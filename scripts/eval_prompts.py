"""Evaluate KG extraction prompts against golden test cases.

Runs the GraphRAGExtractor against golden test cases and computes
P/R/F1 metrics for entity and relationship extraction quality.

Supports all four extraction modes:
  - single-pass       (legacy parse_fn)
  - two-pass          (entities then relationships)
  - instructor        (Pydantic-validated output)
  - two-pass-instructor (both)

Usage:
    python scripts/eval_prompts.py --mode single-pass
    python scripts/eval_prompts.py --mode two-pass
    python scripts/eval_prompts.py --mode instructor
    python scripts/eval_prompts.py --mode two-pass-instructor
    python scripts/eval_prompts.py --mode single-pass --output results/baseline.json
    python scripts/eval_prompts.py --mode single-pass --verbose   # show FP/FN details
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add project src/ to sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))  # for tests.eval.harness

# Llama_index has a heavy C dependency (numpy) that can crash on NixOS.
# Mock it early if we only need the extractor's core logic.
# The eval script calls the extractor directly, so we need it — but
# we'll handle ImportError gracefully below.

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.llms.openai_like import OpenAILike

from config import get_config
from core.prompts import PromptRegistry
from core.extractor import GraphRAGExtractor

# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------
from tests.eval.harness import EvaluationHarness, GoldenCase, AggregateResult

# ---------------------------------------------------------------------------
# JSON parse function for single-pass mode
# ---------------------------------------------------------------------------
# The kg_extract_template.txt prompt returns structured JSON, not the
# "(subject, predicate, object)" triplet format that LlamaIndex's
# default_parse_triplets_fn expects.  We must supply a custom parse_fn
# that extracts (name, type, description) entities and
# (source, target, relation, description) relationships from JSON.
# ---------------------------------------------------------------------------

import json as _json
import re as _re


def _extract_json(text: str):
    """Extract and parse JSON from LLM output text.

    Handles common LLM output issues:
    - Double braces {{ }} from PromptTemplate escape sequences
    - Markdown code fences (```json ... ```)
    - Partial or truncated JSON at the end

    First normalizes double braces, then tries a fast regex match,
    then falls back to progressively shrinking the substring.
    """
    # Normalize double braces from PromptTemplate escape syntax.
    # LLMs may echo {{ }} from prompt examples that weren't properly
    # formatted through PromptTemplate.format().
    text = text.replace("{{", "{").replace("}}", "}")

    # Try to extract JSON from markdown code blocks first
    md_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if md_match:
        try:
            return _json.loads(md_match.group(1))
        except _json.JSONDecodeError:
            pass

    match = _re.search(r"\{.*\}", text, _re.DOTALL)
    if match:
        try:
            return _json.loads(match.group(0))
        except _json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        return None
    for end in range(len(text), start, -1):
        substring = text[start:end]
        try:
            return _json.loads(substring)
        except _json.JSONDecodeError:
            continue
    return None


def kg_parse_fn(response_str: str):
    """Parse LLM JSON response into (entities, relationships) tuples.

    Returns
    -------
    entities : list of (name, type, description)
    relationships : list of (source, target, relation, description)
    """
    logger.debug("kg_parse_fn received response (first 500 chars): %s", response_str[:500])
    entities = []
    relationships = []
    data = _extract_json(response_str)
    if not data or not isinstance(data, dict):
        logger.warning("kg_parse_fn: _extract_json returned %s", type(data).__name__ if data else "None")
        return entities, relationships
    try:
        entities = [
            (
                entity["entity_name"],
                entity["entity_type"],
                entity["entity_description"],
            )
            for entity in data.get("entities", [])
        ]
    except (KeyError, TypeError):
        pass
    try:
        relationships = [
            (
                rel["source_entity"],
                rel["target_entity"],
                rel["relation"],
                rel["relationship_description"],
            )
            for rel in data.get("relationships", [])
        ]
    except (KeyError, TypeError):
        pass
    return entities, relationships

logger = logging.getLogger("eval_prompts")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def setup_llm(config) -> OpenAILike:
    """Initialize the LLM from config (Groq or OpenAI-compatible)."""
    if not config.llm.api_key:
        raise SystemExit(
            "OPENAI_API_KEY environment variable is required. "
            "Set it with: export OPENAI_API_KEY=sk-..."
        )
    llm = OpenAILike(
        model=config.llm.model,
        api_base=config.llm.api_base,
        api_key=config.llm.api_key,
        is_chat_model=True,
    )
    Settings.llm = llm
    logger.info("LLM: %s @ %s", config.llm.model, config.llm.api_base)
    return llm


def setup_instructor_client(config):
    """Create an Instructor client for structured extraction modes."""
    try:
        import instructor
        import openai
    except ImportError:
        raise SystemExit(
            "The 'instructor' and 'openai' packages are required for "
            "instructor modes. Install with: pip install instructor openai"
        )
    client = instructor.from_openai(
        openai.OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.api_base,
        ),
    )
    return client


# ---------------------------------------------------------------------------
# Extraction function
# ---------------------------------------------------------------------------

async def run_extraction(
    text: str,
    mode: str,
    llm: OpenAILike,
    config,
    reg: PromptRegistry,
    instructor_client=None,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]]]:
    """Run extraction on a text string, return (entities, relationships).

    Parameters
    ----------
    text : str
        The document text to extract from.
    mode : str
        One of 'single-pass', 'two-pass', 'instructor', 'two-pass-instructor'.
    llm : OpenAILike
        The LlamaIndex LLM instance.
    config : Config
        The Study Buddy config object.
    reg : PromptRegistry
        Loaded prompt registry.
    instructor_client : optional
        Instructor client (required for instructor modes).

    Returns
    -------
    entities : list of (name, type, description)
    relationships : list of (source, target, relation, description)
    """
    use_instructor = mode in ("instructor", "two-pass-instructor")
    use_two_pass = mode in ("two-pass", "two-pass-instructor")

    # Build extractor kwargs
    extractor_kwargs = {
        "llm": llm,
        "extract_prompt": reg.raw("kg_extract"),
        "max_paths_per_chunk": config.graphrag.max_paths_per_chunk,
        "use_instructor": use_instructor,
        "use_two_pass": use_two_pass,
    }

    # Single-pass mode needs a custom parse_fn because the prompt returns
    # JSON (not LlamaIndex's expected "(subj, pred, obj)" triplets).
    # Two-pass and instructor modes parse JSON internally.
    if not use_two_pass and not use_instructor:
        extractor_kwargs["parse_fn"] = kg_parse_fn

    if use_instructor:
        if instructor_client is None:
            instructor_client = setup_instructor_client(config)
        extractor_kwargs["instructor_client"] = instructor_client
        extractor_kwargs["instructor_model_name"] = config.llm.model
        extractor_kwargs["instructor_max_retries"] = config.graphrag.instructor_max_retries

    if use_two_pass:
        extractor_kwargs["entity_prompt"] = reg.raw("kg_extract_entities")
        extractor_kwargs["relationship_prompt"] = reg.raw("kg_extract_relationships")

    extractor = GraphRAGExtractor(**extractor_kwargs)
    logger.debug("    Extractor created: parse_fn=%s", getattr(extractor, 'parse_fn', None).__name__ if hasattr(extractor, 'parse_fn') else 'default')

    # Create a TextNode from the input text
    node = TextNode(text=text, id_="eval-node")

    # Run extraction — use acall() which routes to the correct method
    # based on use_two_pass / use_instructor flags set on the extractor
    try:
        result_nodes = await extractor.acall([node])
        result_node = result_nodes[0]
    except Exception as e:
        logger.error("Extraction failed for text (first 80 chars): %s...", text[:80])
        logger.error("Error: %s: %s", type(e).__name__, e)
        return [], []

    # Parse result from node metadata
    entities = []
    relationships = []

    # LlamaIndex uses these keys in node metadata
    kg_nodes = result_node.metadata.get("nodes", [])
    kg_rels = result_node.metadata.get("relations", [])

    # Debug: log raw metadata keys and counts
    logger.debug("    Result metadata keys: %s", list(result_node.metadata.keys()))
    logger.debug("    kg_nodes type: %s, count: %d", type(kg_nodes).__name__, len(kg_nodes) if isinstance(kg_nodes, list) else 'N/A')
    logger.debug("    kg_rels type: %s, count: %d", type(kg_rels).__name__, len(kg_rels) if isinstance(kg_rels, list) else 'N/A')

    for entity_node in kg_nodes:
        name = getattr(entity_node, "name", "")
        label = getattr(entity_node, "label", "")
        desc = entity_node.properties.get("entity_description", "") if hasattr(entity_node, "properties") else ""
        entities.append((name, label, desc))

    for rel in kg_rels:
        source = getattr(rel, "source_id", "")
        target = getattr(rel, "target_id", "")
        label = getattr(rel, "label", "")
        desc = rel.properties.get("relationship_description", "") if hasattr(rel, "properties") else ""
        relationships.append((source, target, label, desc))

    logger.debug("    Extracted %d entities, %d relationships from node metadata", len(entities), len(relationships))
    if logger.isEnabledFor(logging.DEBUG) and entities:
        logger.debug("    First 3 entities: %s", entities[:3])
        logger.debug("    First 3 relationships: %s", relationships[:3])

    return entities, relationships


def extract_sync(
    text: str,
    mode: str,
    llm: OpenAILike,
    config,
    reg: PromptRegistry,
    instructor_client=None,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]]]:
    """Synchronous wrapper for run_extraction."""
    return asyncio.run(
        run_extraction(text, mode, llm, config, reg, instructor_client)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KG extraction prompts against golden test cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_prompts.py --mode single-pass
  python scripts/eval_prompts.py --mode two-pass-instructor --output results/combined.json
  python scripts/eval_prompts.py --mode single-pass --verbose
        """,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["single-pass", "two-pass", "instructor", "two-pass-instructor"],
        help="Extraction mode to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="results/eval.json",
        help="Path to write results JSON (default: results/eval.json).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-case evaluation details.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw LLM responses and parsed results for debugging.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between API calls in seconds (default: 2.0). Helps with rate limits.",
    )
    args = parser.parse_args()

    # Load config
    config = get_config()

    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Setup LLM
    llm = setup_llm(config)

    # Setup prompts
    reg = PromptRegistry(config=config.graphrag)

    # Setup instructor client if needed
    instructor_client = None
    if args.mode in ("instructor", "two-pass-instructor"):
        instructor_client = setup_instructor_client(config)

    # Load golden cases
    cases = GoldenCase.load_all()
    if not cases:
        logger.error("No golden cases found. Add JSON files to tests/eval/golden/.")
        sys.exit(1)
    logger.info("Loaded %d golden cases: %s", len(cases), [c.key for c in cases])

    # Run eval
    harness = EvaluationHarness()
    results = []

    logger.info("Running eval in '%s' mode with model %s", args.mode, config.llm.model)
    start_time = time.time()

    for i, case in enumerate(cases):
        logger.info(
            "  [%d/%d] Extracting: %s (%d chars)...",
            i + 1, len(cases), case.key, len(case.text),
        )

        try:
            predicted_entities, predicted_rels = extract_sync(
                text=case.text,
                mode=args.mode,
                llm=llm,
                config=config,
                reg=reg,
                instructor_client=instructor_client,
            )
        except Exception as e:
            logger.error("  FAILED: %s: %s", type(e).__name__, e)
            continue

        result = harness.evaluate(
            predicted_entities=predicted_entities,
            predicted_relationships=predicted_rels,
            golden_key=case.key,
        )
        results.append(result)

        if args.verbose:
            print(result.summary())
            print()
        else:
            logger.info(
                "  Entities: P=%.2f%% R=%.2f%% F1=%.2f%%  Rels: P=%.2f%% R=%.2f%% F1=%.2f%%",
                result.entity_metrics.precision * 100,
                result.entity_metrics.recall * 100,
                result.entity_metrics.f1 * 100,
                result.relationship_metrics.precision * 100,
                result.relationship_metrics.recall * 100,
                result.relationship_metrics.f1 * 100,
            )

        # Rate limit protection
        if i < len(cases) - 1:
            time.sleep(args.delay)

    if not results:
        logger.error("No results collected. Check LLM configuration and golden cases.")
        sys.exit(1)

    # Compute aggregate
    n = len(results)
    aggregate = AggregateResult(
        num_cases=n,
        entity_precision=sum(r.entity_metrics.precision for r in results) / n,
        entity_recall=sum(r.entity_metrics.recall for r in results) / n,
        entity_f1=sum(r.entity_metrics.f1 for r in results) / n,
        entity_name_precision=sum(r.entity_name_metrics.precision for r in results if r.entity_name_metrics) / n,
        entity_name_recall=sum(r.entity_name_metrics.recall for r in results if r.entity_name_metrics) / n,
        entity_name_f1=sum(r.entity_name_metrics.f1 for r in results if r.entity_name_metrics) / n,
        relationship_precision=sum(r.relationship_metrics.precision for r in results) / n,
        relationship_recall=sum(r.relationship_metrics.recall for r in results) / n,
        relationship_f1=sum(r.relationship_metrics.f1 for r in results) / n,
        rel_fuzzy_precision=sum(r.rel_fuzzy_metrics.precision for r in results if r.rel_fuzzy_metrics) / n,
        rel_fuzzy_recall=sum(r.rel_fuzzy_metrics.recall for r in results if r.rel_fuzzy_metrics) / n,
        rel_fuzzy_f1=sum(r.rel_fuzzy_metrics.f1 for r in results if r.rel_fuzzy_metrics) / n,
    )

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 60)
    print(aggregate.summary())
    print("=" * 60)
    print(f"\nMode:      {args.mode}")
    print(f"Model:     {config.llm.model}")
    print(f"API:       {config.llm.api_base}")
    print(f"Cases:     {n}")
    print(f"Time:      {elapsed:.1f}s")

    # Print per-case details in verbose mode
    if args.verbose:
        print("\n--- Per-Case Details ---")
        for r in results:
            print(r.summary())
            print()

    # Save results to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "mode": args.mode,
        "model": config.llm.model,
        "api_base": config.llm.api_base,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "num_cases": n,
        "aggregate": {
            "entity_precision": aggregate.entity_precision,
            "entity_recall": aggregate.entity_recall,
            "entity_f1": aggregate.entity_f1,
            "entity_name_precision": aggregate.entity_name_precision,
            "entity_name_recall": aggregate.entity_name_recall,
            "entity_name_f1": aggregate.entity_name_f1,
            "relationship_precision": aggregate.relationship_precision,
            "relationship_recall": aggregate.relationship_recall,
            "relationship_f1": aggregate.relationship_f1,
            "rel_fuzzy_precision": aggregate.rel_fuzzy_precision,
            "rel_fuzzy_recall": aggregate.rel_fuzzy_recall,
            "rel_fuzzy_f1": aggregate.rel_fuzzy_f1,
        },
        "cases": [
            {
                "golden_key": r.golden_key,
                "entity_precision": r.entity_metrics.precision,
                "entity_recall": r.entity_metrics.recall,
                "entity_f1": r.entity_metrics.f1,
                "entity_matched": r.entity_metrics.matched_count,
                "entity_golden": r.entity_metrics.golden_count,
                "entity_predicted": r.entity_metrics.predicted_count,
                "entity_name_precision": r.entity_name_metrics.precision if r.entity_name_metrics else None,
                "entity_name_recall": r.entity_name_metrics.recall if r.entity_name_metrics else None,
                "entity_name_f1": r.entity_name_metrics.f1 if r.entity_name_metrics else None,
                "relationship_precision": r.relationship_metrics.precision,
                "relationship_recall": r.relationship_metrics.recall,
                "relationship_f1": r.relationship_metrics.f1,
                "relationship_matched": r.relationship_metrics.matched_count,
                "relationship_golden": r.relationship_metrics.golden_count,
                "relationship_predicted": r.relationship_metrics.predicted_count,
                "rel_fuzzy_precision": r.rel_fuzzy_metrics.precision if r.rel_fuzzy_metrics else None,
                "rel_fuzzy_recall": r.rel_fuzzy_metrics.recall if r.rel_fuzzy_metrics else None,
                "rel_fuzzy_f1": r.rel_fuzzy_metrics.f1 if r.rel_fuzzy_metrics else None,
            }
            for r in results
        ],
    }

    output_path.write_text(json.dumps(output_data, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print suggestion for updating frontmatter
    print(f"\nTo update prompt frontmatter, set eval_score to: {aggregate.entity_f1:.4f}")
    print(f"  Edit src/prompts/kg_extract_template.txt and set eval_score: {aggregate.entity_f1:.4f}")


if __name__ == "__main__":
    main()