"""Unit tests for select_publications."""

from src.matching.select_publications import (
    compute_publication_stats,
    load_references,
    read_id_file,
    select_publications,
)


def _entries(ws, n):
    return [ws.entry(f"k{i}", title=f"Title number {i}") for i in range(n)]


def _refs(ws, n):
    return [ws.ref(f"2001.{i:05d}", f"Ref title {i}") for i in range(n)]


def test_load_references_keeps_only_arxiv_and_dedups(workspace) -> None:
    refs = [
        workspace.ref("2001.00001", "A"),
        workspace.ref("2001.00001", "A duplicate"),
        workspace.ref(None, "Book without arXiv id"),
    ]
    workspace.add_publication("2512.00001", [], refs)
    loaded = load_references("2512.00001", workspace.corpus_dir)
    assert [r["ref_arxiv_id"] for r in loaded] == ["2001.00001"]


def test_stats_and_selection(workspace) -> None:
    workspace.add_publication("2512.00001", _entries(workspace, 25), _refs(workspace, 12))  # eligible
    workspace.add_publication("2512.00002", _entries(workspace, 10), _refs(workspace, 12))  # too few entries
    workspace.add_publication("2512.00003", _entries(workspace, 25), _refs(workspace, 3))   # too few refs
    workspace.add_publication("2512.00004", _entries(workspace, 30), _refs(workspace, 20))  # best

    ids = ["2512.00001", "2512.00002", "2512.00003", "2512.00004"]
    stats = compute_publication_stats(ids, workspace.intermediate_dir, workspace.corpus_dir)
    by_id = {s["arxiv_id"]: s for s in stats}
    assert by_id["2512.00001"]["n_bibtex"] == 25
    assert by_id["2512.00001"]["expected_matchable"] == 12

    selected = select_publications(stats, num=5, min_bibtex=20, min_arxiv_refs=10)
    assert [s["arxiv_id"] for s in selected] == ["2512.00004", "2512.00001"]


def test_read_id_file_ignores_comments(tmp_path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("# comment\n2512.00001\n\n2512.00002\n", encoding="utf-8")
    assert read_id_file(path) == ["2512.00001", "2512.00002"]
    assert read_id_file(tmp_path / "missing.txt") == []