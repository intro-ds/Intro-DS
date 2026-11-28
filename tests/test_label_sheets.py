from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.matching.label_sheets import SHEET_COLUMNS, generate_label_sheets


def test_generate_label_sheets_top3_and_empty_groundtruth(tmp_path: Path):
    bibtex_path = tmp_path / "bibtex.parquet"
    refs_path = tmp_path / "references.parquet"
    out_dir = tmp_path / "sheets"

    pd.DataFrame([
        {
            "arxiv_id": "2607.23694",
            "citation_key": "vaswani2017",
            "entry_type": "inproceedings",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "doi": "10.5555/3295222.3295349",
            "journal": None,
            "url": None,
        },
        {
            "arxiv_id": "2607.23694",
            "citation_key": "unmatched2020",
            "entry_type": "article",
            "title": "Completely Unrelated Topic",
            "authors": ["Jane Doe"],
            "year": 2020,
            "doi": None,
            "journal": None,
            "url": None,
        },
    ]).to_parquet(bibtex_path, index=False)

    pd.DataFrame([
        {
            "arxiv_id": "2607.23694",
            "ref_arxiv_id": "1706.03762",
            "ref_title": "Attention Is All You Need",
            "ref_authors": ["Ashish Vaswani"],
            "ref_submission_date": "2017-06-12",
            "ref_s2_id": "s2_1",
        },
        {
            "arxiv_id": "2607.23694",
            "ref_arxiv_id": "1810.04805",
            "ref_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "ref_authors": ["Jacob Devlin"],
            "ref_submission_date": "2018-10-11",
            "ref_s2_id": "s2_2",
        },
        {
            "arxiv_id": "2607.23694",
            "ref_arxiv_id": "2005.14165",
            "ref_title": "Language Models are Few-Shot Learners",
            "ref_authors": ["Tom Brown"],
            "ref_submission_date": "2020-05-28",
            "ref_s2_id": "s2_3",
        },
        {
            "arxiv_id": "2607.23694",
            "ref_arxiv_id": "1512.03385",
            "ref_title": "Deep Residual Learning for Image Recognition",
            "ref_authors": ["Kaiming He"],
            "ref_submission_date": "2015-12-10",
            "ref_s2_id": "s2_4",
        },
    ]).to_parquet(refs_path, index=False)

    paths = generate_label_sheets(
        arxiv_ids=["2607.23694"],
        bibtex_path=bibtex_path,
        references_path=refs_path,
        output_dir=out_dir,
    )

    assert len(paths) == 1
    assert paths[0] == out_dir / "2607.23694.csv"

    with paths[0].open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SHEET_COLUMNS
        rows = list(reader)

    assert len(rows) == 2
    first = rows[0]
    assert first["key"] == "vaswani2017"
    assert first["authors"] == "Ashish Vaswani; Noam Shazeer"
    assert first["year"] == "2017"
    assert first["groundtruth_arxiv_id"] == ""
    assert first["cand_1_arxiv_id"] == "1706.03762"
    assert float(first["cand_1_score"]) == 1.0
    assert first["cand_2_arxiv_id"] != ""
    assert first["cand_3_arxiv_id"] != ""