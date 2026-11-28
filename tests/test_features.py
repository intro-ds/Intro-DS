from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.matching.features import FEATURE_COLUMNS, build_pair_features


def test_build_pair_features_matrix_and_relative_ranks(tmp_path: Path):
    bibtex_path = tmp_path / "bibtex.parquet"
    refs_path = tmp_path / "references.parquet"

    pd.DataFrame([
        {
            "arxiv_id": "2607.23694",
            "citation_key": "vaswani2017",
            "entry_type": "inproceedings",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "doi": "10.48550/arXiv.1706.03762",
            "journal": None,
            "url": "https://arxiv.org/abs/1706.03762",
        },
        {
            "arxiv_id": "2607.23694",
            "citation_key": "devlin2018",
            "entry_type": "article",
            "title": "BERT Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "year": 2018,
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
            "ref_authors": ["Ashish Vaswani", "Noam Shazeer"],
            "ref_submission_date": "2017-06-12",
            "ref_s2_id": "s2_1",
        },
        {
            "arxiv_id": "2607.23694",
            "ref_arxiv_id": "1810.04805",
            "ref_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "ref_authors": ["Jacob Devlin"],
            "ref_submission_date": "2019-05-24",
            "ref_s2_id": "s2_2",
        },
    ]).to_parquet(refs_path, index=False)

    df = build_pair_features(
        arxiv_id="2607.23694",
        bibtex_path=bibtex_path,
        references_path=refs_path,
    )

    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 4

    pos_vaswani = df[
        (df["citation_key"] == "vaswani2017") & (df["ref_arxiv_id"] == "1706.03762")
    ].iloc[0]
    neg_vaswani = df[
        (df["citation_key"] == "vaswani2017") & (df["ref_arxiv_id"] == "1810.04805")
    ].iloc[0]

    assert pos_vaswani["title_jaccard"] == 1.0
    assert pos_vaswani["title_char_tfidf_cosine"] > 0.99
    assert pos_vaswani["title_fuzz_ratio"] == 1.0
    assert pos_vaswani["title_len_diff"] == 0
    assert pos_vaswani["first_author_last_name_match"] == 1
    assert pos_vaswani["author_last_names_jaccard"] == 1.0
    assert pos_vaswani["year_diff"] == 0.0
    assert pos_vaswani["year_exact_match"] == 1
    assert pos_vaswani["doi_match"] == 1
    assert pos_vaswani["arxiv_id_match"] == 1
    assert pos_vaswani["title_jaccard_rank"] == 1
    assert pos_vaswani["title_jaccard_diff_from_best"] == 0.0
    assert neg_vaswani["title_jaccard_rank"] == 2
    assert neg_vaswani["title_jaccard_diff_from_best"] > 0.0


def test_build_pair_features_empty_publication(tmp_path: Path):
    bibtex_path = tmp_path / "bibtex.parquet"
    refs_path = tmp_path / "references.parquet"
    pd.DataFrame(columns=["arxiv_id", "citation_key", "title"]).to_parquet(bibtex_path, index=False)
    pd.DataFrame(columns=["arxiv_id", "ref_arxiv_id", "ref_title"]).to_parquet(refs_path, index=False)

    df = build_pair_features("9999.99999", bibtex_path=bibtex_path, references_path=refs_path)
    assert df.empty
    assert list(df.columns) == FEATURE_COLUMNS