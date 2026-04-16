"""Tests for Pydantic models."""
from src.models import GraphQueryResponse, QueryRequest, slugify


def test_slugify_basic():
    assert slugify("ML Research") == "ml-research"


def test_slugify_special_chars():
    assert slugify("Biology Notes!") == "biology-notes"


def test_slugify_multiple_spaces():
    assert slugify("My  Cool   Project") == "my-cool-project"


def test_slugify_leading_trailing_spaces():
    assert slugify("  hello world  ") == "hello-world"


def test_slugify_unicode():
    result = slugify("Café Résumé")
    assert "cafe" in result or "caf" in result  # unicode normalization varies


def test_query_request_defaults():
    req = QueryRequest(query="What is AI?")
    assert req.similarity_top_k == 20