"""Unit tests for the deduplicator module."""

import pytest

from src.latexParser.deduplicator import deduplicate_bibtex, deduplicate_elements
from src.latexParser.hierarchy_builder import (
    KIND_DOCUMENT,
    KIND_HEADING,
    KIND_SENTENCE,
    Element,
)


def test_element_deduplication_across_versions() -> None:
    """Test that identical elements across versions share a single element_id."""
    # Version 1 tree
    v1_root = Element(
        element_id="2512-00001#root1",
        kind=KIND_DOCUMENT,
        content="2512.00001",
        children=[
            Element(
                element_id="2512-00001#h1",
                kind=KIND_HEADING,
                content="Introduction",
                children=[
                    Element(
                        element_id="2512-00001#s1",
                        kind=KIND_SENTENCE,
                        content="We study the citation matching problem.",
                        children=[],
                    )
                ],
            )
        ],
    )

    # Version 2 tree: different heading, identical sentence
    v2_root = Element(
        element_id="2512-00001#root2",
        kind=KIND_DOCUMENT,
        content="2512.00001",
        children=[
            Element(
                element_id="2512-00001#h2",
                kind=KIND_HEADING,
                content="Background and Model",
                children=[
                    Element(
                        element_id="2512-00001#s2",
                        kind=KIND_SENTENCE,
                        content="We study the citation matching problem.",
                        children=[],
                    )
                ],
            )
        ],
    )

    version_trees = {1: v1_root, 2: v2_root}
    elements_rows, hierarchy_rows = deduplicate_elements(version_trees, "2512.00001")

    # In elements_rows, the duplicate sentence should appear only ONCE
    sentences = [e for e in elements_rows if e["kind"] == KIND_SENTENCE]
    assert len(sentences) == 1
    shared_id = sentences[0]["element_id"]
    assert shared_id == "2512-00001#s1"

    # In hierarchy_rows, version 1 has s1 under h1, version 2 has s1 under h2
    v1_edge = [h for h in hierarchy_rows if h["version"] == 1 and h["element_id"] == shared_id]
    v2_edge = [h for h in hierarchy_rows if h["version"] == 2 and h["element_id"] == shared_id]

    assert len(v1_edge) == 1
    assert v1_edge[0]["parent_id"] == "2512-00001#h1"

    assert len(v2_edge) == 1
    assert v2_edge[0]["parent_id"] == "2512-00001#h2"


def test_bibtex_deduplication_and_field_union() -> None:
    """Test deduplicating duplicate BibTeX entries and unionizing their fields."""
    v1_bib = [
        {
            "citation_key": "vaswani2017",
            "entry_type": "article",
            "title": "Attention Is All You Need",
            "authors": ["A. Vaswani", "N. Shazeer"],
            "year": 2017,
            "doi": None,
            "journal": "NeurIPS",
            "url": None,
        }
    ]

    v2_bib = [
        {
            "citation_key": "vit_transformer",
            "entry_type": "article",
            "title": "Attention Is All You Need",
            "authors": ["A. Vaswani", "N. Shazeer", "N. Parmar"],
            "year": None,
            "doi": "10.1234/neurips.2017",
            "journal": None,
            "url": "https://arxiv.org/abs/1706.03762",
        }
    ]

    version_bibtex = {1: v1_bib, 2: v2_bib}
    bibtex_rows, renames = deduplicate_bibtex(version_bibtex, "2512.00001")

    # Only one unified entry should exist
    assert len(bibtex_rows) == 1
    unified = bibtex_rows[0]

    assert unified["citation_key"] == "vaswani2017"
    assert unified["title"] == "Attention Is All You Need"
    assert unified["year"] == 2017
    assert unified["doi"] == "10.1234/neurips.2017"
    assert unified["journal"] == "NeurIPS"
    assert unified["url"] == "https://arxiv.org/abs/1706.03762"
    assert len(unified["authors"]) >= 2

    # Renames should map vit_transformer -> vaswani2017
    assert renames.get("vit_transformer") == "vaswani2017"
