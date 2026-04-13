"""
Shared parsing utilities for the Study Buddy application.

Consolidates JSON extraction and entity/relationship parsing
that was previously duplicated between app.py and main.py.
"""

import json
import re
from typing import List, Tuple


def extract_json(text: str):
    """
    Extract and parse JSON from text.

    First tries a fast regex match, then falls back to progressively
    shrinking the substring from the end until valid JSON is found.

    Returns parsed dict on success, None on failure.
    """
    # Fast path: try regex match first
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass  # Fall through to slow path

    # Slow path: find first { and shrink from end
    start = text.find("{")
    if start == -1:
        return None

    for end in range(len(text), start, -1):
        substring = text[start:end]
        try:
            return json.loads(substring)
        except json.JSONDecodeError:
            continue

    return None


def parse_fn(response_str: str) -> Tuple[List, List]:
    """Parse LLM response for entity/relationship extraction.

    Returns:
        Tuple of (entities, relationships) where:
        - entities: List of (name, type, description) tuples
        - relationships: List of (source, target, relation, description) tuples
    """
    entities: List = []
    relationships: List = []
    data = extract_json(response_str)
    if not data:
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
        relationships = [
            (
                relation["source_entity"],
                relation["target_entity"],
                relation["relation"],
                relation["relationship_description"],
            )
            for relation in data.get("relationships", [])
        ]
        return entities, relationships
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        import logging
        logging.getLogger(__name__).warning(f"Error parsing LLM JSON response: {e}")
        return [], []


# Supported file extensions for document ingestion
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".html", ".xlsx",
    ".md", ".csv", ".txt", ".json",
}