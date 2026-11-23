"""Unit tests for auto_labeler."""

import csv

from src.matching.auto_labeler import (
    label_entry,
    label_publication,
    normalize_text,
    run_auto_labeling,
    title_similarity,
)

REFS = [
    {
        "ref_arxiv_id": "1706.03762",
        "ref_title": "Attention is All you Need",
        "ref_authors": ["Ashish Vaswani", "Noam Shazeer"],
        "ref_submission_date": "2017-06-12",
    },
    {
        "ref_arxiv_id": "1412.6980",
        "ref_title": "Adam: A Method for Stochastic Optimization",
        "ref_authors": ["Diederik Kingma", "Jimmy Ba"],
        "ref_submission_date": "2014-12-22",
    },
]


def test_normalize_text_strips_accents_and_latex() -> None:
    assert normalize_text(r"Schr\"odinger's \emph{Cat}: Z\u00fcrich") != ""
    assert normalize_text("Café  Déjà-Vu!") == "cafe deja vu"
    assert normalize_text(None) == ""


def test_title_similarity() -> None:
    assert title_similarity("Attention Is All You Need", "attention is all you need.") > 0.95
    assert title_similarity("Quantum error correction codes", "Attention Is All You Need") < 0.3
    assert title_similarity("", "Anything") is None


def test_label_entry_by_title_authors_year(workspace) -> None:
    entry = workspace.entry("vaswani2017", "Attention Is All You Need", ["A. Vaswani", "N. Shazeer"], 2017)
    ref_id, score, method = label_entry(entry, REFS)
    assert (ref_id, method) == ("1706.03762", "title_authors")
    assert score >= 0.8


def test_label_entry_by_arxiv_id_in_url(workspace) -> None:
    entry = workspace.entry("adam", None, [], None, url="https://arxiv.org/abs/1412.6980v9")
    ref_id, score, method = label_entry(entry, REFS)
    assert (ref_id, score, method) == ("1412.6980", 1.0, "arxiv_id")


def test_label_entry_no_match_and_no_candidates(workspace) -> None:
    entry = workspace.entry("x", "Quantum error correction codes", ["J. Preskill"], 2018)
    assert label_entry(entry, REFS)[0] is None
    assert label_entry(entry, [])[2] == "no_candidates"


def test_label_publication_labels_every_entry(workspace) -> None:
    entries = [
        workspace.entry("a", "Attention Is All You Need", ["Vaswani"], 2017),
        workspace.entry("b", "Unrelated survey of graph databases", ["Smith"], 2020),
    ]
    rows = label_publication("2512.00001", entries, REFS)
    assert len(rows) == 2
    assert rows[0]["groundtruth_arxiv_id"] == "1706.03762"
    assert rows[1]["groundtruth_arxiv_id"] == ""
    assert all(r["source"] == "auto" for r in rows)


def test_run_auto_labeling_excludes_manual_and_hits_fraction(workspace, matched_publication) -> None:
    ids = [f"2511.{i:05d}" for i in range(1, 11)]
    for aid in ids:
        matched_publication(aid)

    summary = run_auto_labeling(
        workspace.intermediate_dir,
        workspace.corpus_dir,
        workspace.output_dir,
        manual_ids={ids[0]},
        fraction=0.2,
        min_publications=2,
        min_bibtex=5,
    )
    assert summary["pool_size"] == 9
    assert summary["required_publications"] == 2  # ceil(0.2 * 9) = 2
    assert summary["selected_publications"] == 2
    assert summary["requirement_met"] is True
    assert summary["entries_labelled"] == 10  # 2 publications x 5 entries

    with (workspace.output_dir / "labels_auto.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert ids[0] not in {r["arxiv_id"] for r in rows}
    assert summary["match_rate"] == 1.0