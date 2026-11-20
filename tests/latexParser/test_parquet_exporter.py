"""Unit tests for the parquet_exporter module."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.latexParser.parquet_exporter import export_parquet_tables


def test_export_parquet_tables(tmp_path: Path) -> None:
    """Test exporting elements, hierarchy, and bibtex Parquet tables."""
    intermediate_dir = tmp_path / "intermediate"
    processed_dir = tmp_path / "processed"
    paper_dir = intermediate_dir / "2512.00001"
    paper_dir.mkdir(parents=True)

    dummy_data = {
        "arxiv_id": "2512.00001",
        "versions": {"1": {"parsed": True, "element_count": 2, "bib_count": 1}},
        "elements": [
            {
                "element_id": "2512-00001#root1",
                "arxiv_id": "2512.00001",
                "kind": "document",
                "content": "2512.00001",
            },
            {
                "element_id": "2512-00001#s1",
                "arxiv_id": "2512.00001",
                "kind": "sentence",
                "content": "Sample content sentence.",
            },
        ],
        "hierarchy": [
            {
                "arxiv_id": "2512.00001",
                "version": 1,
                "element_id": "2512-00001#s1",
                "parent_id": "2512-00001#root1",
            }
        ],
        "bibtex": [
            {
                "arxiv_id": "2512.00001",
                "citation_key": "vaswani2017",
                "entry_type": "article",
                "title": "Attention Is All You Need",
                "authors": ["A. Vaswani"],
                "year": 2017,
                "doi": "10.1234/test",
                "journal": "NeurIPS",
                "url": "https://arxiv.org/abs/1706.03762",
            }
        ],
    }

    with (paper_dir / "parsed.json").open("w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    res = export_parquet_tables(
        student_id="24127559",
        intermediate_dir=intermediate_dir,
        output_dir=processed_dir,
    )

    assert res["elements_count"] == 2
    assert res["hierarchy_count"] == 1
    assert res["bibtex_count"] == 1

    # Verify elements.parquet
    elem_df = pd.read_parquet(res["elements_path"])
    assert list(elem_df.columns) == ["element_id", "arxiv_id", "kind", "content"]
    assert len(elem_df) == 2
    assert elem_df["element_id"].iloc[0] == "2512-00001#root1"

    # Verify hierarchy.parquet
    hier_df = pd.read_parquet(res["hierarchy_path"])
    assert list(hier_df.columns) == ["arxiv_id", "version", "element_id", "parent_id"]
    assert len(hier_df) == 1
    assert hier_df["version"].iloc[0] == 1

    # Verify bibtex.parquet
    bib_df = pd.read_parquet(res["bibtex_path"])
    assert list(bib_df.columns) == [
        "arxiv_id",
        "citation_key",
        "entry_type",
        "title",
        "authors",
        "year",
        "doi",
        "journal",
        "url",
    ]
    assert len(bib_df) == 1
    assert bib_df["citation_key"].iloc[0] == "vaswani2017"
    assert bib_df["year"].iloc[0] == 2017
