import json

import pandas as pd
import pytest

import src.matching.auto_matcher as matcher


def test_auto_label_matches_by_id_doi_title_and_fuzzy_title_and_labels_every_entry(
    tmp_path, monkeypatch
):
    papers_path = tmp_path / "papers.parquet"
    metadata_path = tmp_path / "metadata.jsonl"
    semantic_scholar_dir = tmp_path / "semantic_scholar"
    semantic_scholar_dir.mkdir()
    citing_id = "2603.10580"

    pd.DataFrame([
        {
            "arxiv_id": "2501.11111",
            "title": "ArXiv ID target",
            "authors": ["Alice Smith"],
            "submission_date": "2021-01-01",
        },
        {
            "arxiv_id": "2501.22222",
            "title": "DOI target",
            "authors": ["Bob Jones"],
            "submission_date": "2022-01-01",
        },
        {
            "arxiv_id": "2501.33333",
            "title": "Normalized Title for Matching",
            "authors": ["Carol Brown"],
            "submission_date": "2023-01-01",
        },
        {
            "arxiv_id": "2501.44444",
            "title": "Robust Sparse Retrieval for Scientific Articles",
            "authors": ["Jane Doe"],
            "submission_date": "2024-01-01",
        },
    ]).to_parquet(papers_path, index=False)
    metadata_path.write_text(
        json.dumps({"id": "2501.22222v1", "doi": "10.1000/doi-target"}) + "\n",
        encoding="utf-8",
    )

    references = [
        {
            "title": "Different title is fine for exact ID",
            "externalIds": {"ArXiv": "2501.11111"},
        },
        {
            "title": "DOI reference",
            "externalIds": {"DOI": "https://doi.org/10.1000/DOI-TARGET"},
        },
        {
            "title": "normalized: TITLE for matching!",
            "externalIds": {},
        },
        {
            "title": "Robust Sparse Retrieval for Scientific Article",
            "year": 2024,
            "authors": [{"name": "Jane Doe"}],
            "externalIds": {},
        },
        {
            "title": "A wholly unrelated reference",
            "year": 2024,
            "authors": [{"name": "Jane Doe"}],
            "externalIds": {},
        },
        {
            "title": "Robust Sparse Retrieval for Scientific Article",
            "year": 2024,
            "authors": [{"name": "Wrong Author"}],
            "externalIds": {},
        },
        {
            "title": "Robust Sparse Retrieval for Scientific Article",
            "year": 2025,
            "authors": [{"name": "Jane Doe"}],
            "externalIds": {},
        },
    ]
    source_path = semantic_scholar_dir / f"{citing_id}.json"
    source_path.write_text(
        json.dumps({"arxiv_id": citing_id, "references": references}),
        encoding="utf-8",
    )

    monkeypatch.setattr(matcher, "PAPERS_PATH", papers_path)
    monkeypatch.setattr(matcher, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(matcher, "SEMANTIC_SCHOLAR_DIR", semantic_scholar_dir)

    labels = matcher.auto_label(citing_id)

    assert len(labels) == len(references)
    assert [entry["label"] for entry in labels] == [
        "matched", "matched", "matched", "matched", "no_match", "no_match",
        "no_match",
    ]
    assert [entry["match_method"] for entry in labels] == [
        "arxiv_id", "doi", "title", "fuzzy_title", None, None, None
    ]
    assert [entry["matched_arxiv_id"] for entry in labels] == [
        "2501.11111", "2501.22222", "2501.33333", "2501.44444", None, None, None
    ]
    assert [entry["reference_index"] for entry in labels] == list(range(7))
    assert labels[-1]["matched_arxiv_id"] is None


def test_auto_label_requires_valid_id_and_reports_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(matcher, "SEMANTIC_SCHOLAR_DIR", tmp_path)

    with pytest.raises(ValueError, match="Invalid arXiv ID"):
        matcher.auto_label("../outside")

    with pytest.raises(FileNotFoundError):
        matcher.auto_label("2603.10580")


def test_auto_label_reads_dois_from_member_metadata_files(tmp_path, monkeypatch):
    papers_path = tmp_path / "papers.parquet"
    pd.DataFrame([{
        "arxiv_id": "2602.16789",
        "title": "A Candidate Paper",
        "authors": ["Alex Author"],
        "submission_date": "2026-02-10",
    }]).to_parquet(papers_path, index=False)
    raw_dir = tmp_path / "data" / "raw"
    member_dir = raw_dir / "Nam"
    member_dir.mkdir(parents=True)
    metadata_path = raw_dir / "arxiv_metadata_all_assignments.jsonl"
    metadata_path.write_text("", encoding="utf-8")
    (member_dir / "arxiv_metadata_2602_all_versions.jsonl").write_text(
        json.dumps({
            "id": "2602.16789v1",
            "doi": "10.1000/member-doi",
        }) + "\n",
        encoding="utf-8",
    )
    semantic_dir = raw_dir / "semantic_scholar"
    semantic_dir.mkdir()
    (semantic_dir / "2603.10580.json").write_text(
        json.dumps({
            "arxiv_id": "2603.10580",
            "references": [{
                "title": "Unrelated citation title",
                "externalIds": {"DOI": "https://doi.org/10.1000/MEMBER-DOI"},
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(matcher, "ROOT", tmp_path)
    monkeypatch.setattr(matcher, "PAPERS_PATH", papers_path)
    monkeypatch.setattr(matcher, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(matcher, "_DEFAULT_METADATA_PATH", metadata_path)
    monkeypatch.setattr(matcher, "SEMANTIC_SCHOLAR_DIR", semantic_dir)

    labels = matcher.auto_label("2603.10580")

    assert labels[0]["label"] == "matched"
    assert labels[0]["matched_arxiv_id"] == "2602.16789"
    assert labels[0]["match_method"] == "doi"
