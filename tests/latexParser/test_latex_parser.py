from __future__ import annotations

import csv
import json
from pathlib import Path
from src.latexParser.file_resolver import LatexFileResolver
from src.latexParser.structure_extractor import PublicationHierarchyBuilder


def _assert_single_parent_per_version(builder: PublicationHierarchyBuilder) -> None:
    element_ids = {row["element_id"] for row in builder.elements_rows}
    edges_by_version_element = {}
    for row in builder.hierarchy_rows:
        key = (row["version"], row["element_id"])
        edges_by_version_element.setdefault(key, []).append(row["parent_id"])

    for parents in edges_by_version_element.values():
        assert len(parents) == 1
        assert len(set(parents)) == 1
        assert parents[0] in element_ids

    for version in {row["version"] for row in builder.hierarchy_rows}:
        parent_by_element = {
            row["element_id"]: row["parent_id"]
            for row in builder.hierarchy_rows
            if row["version"] == version
        }
        for element_id in parent_by_element:
            path = set()
            current = element_id
            while current in parent_by_element:
                assert current not in path
                path.add(current)
                current = parent_by_element[current]


def _parse_document(body: str) -> PublicationHierarchyBuilder:
    builder = PublicationHierarchyBuilder("2310.12345")
    builder.parse_version(1, f"\\begin{{document}}{body}\\end{{document}}")
    return builder

import pyarrow as pa
import pyarrow.parquet as pq

from src.latexParser.run import run_parser


def test_run_parser_two_pass_and_statuses(tmp_path: Path):
    source_dir = tmp_path / "sources"
    v1_dir = source_dir / "2607.23694" / "v1"
    v2_dir = source_dir / "2607.23694" / "v2"
    v3_empty_dir = source_dir / "2607.23695" / "v1"
    v1_dir.mkdir(parents=True)
    v2_dir.mkdir(parents=True)
    v3_empty_dir.mkdir(parents=True)

    (v1_dir / "main.tex").write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Intro}
        See \cite{old_key}.
        \end{document}
        """,
        encoding="utf-8",
    )
    (v2_dir / "main.tex").write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Intro}
        See \cite{new_key}.
        \end{document}
        """,
        encoding="utf-8",
    )

    source_manifest = tmp_path / "source_download_manifest.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arxiv_id", "version", "status"])
        writer.writeheader()
        writer.writerow({"arxiv_id": "2607.23694", "version": "1", "status": "success"})
        writer.writerow({"arxiv_id": "2607.23694", "version": "2", "status": "success"})
        writer.writerow({"arxiv_id": "2607.23695", "version": "1", "status": "no_source"})

    import src.latexParser.run as run_mod

    history_file = tmp_path / "arxiv_history_all_assignments.jsonl"
    history_file.write_text(
        json.dumps({"base_id": "2607.23694", "versions": ["2607.23694v1", "2607.23694v2"]}) + "\n"
        + json.dumps({"base_id": "2607.23695", "versions": ["2607.23695v1"]}) + "\n",
        encoding="utf-8",
    )
    orig_discovery = run_mod.DISCOVERY_FILE
    orig_raw = run_mod.RAW_DIR
    run_mod.DISCOVERY_FILE = history_file
    run_mod.RAW_DIR = tmp_path

    def fake_extract(compiled_tex: str, ver_dir: Path, arxiv_id: str = "", version: int = 1):
        if version == 1:
            return [{"citation_key": "old_key", "entry_type": "article", "title": "Paper A"}]
        return [{"citation_key": "new_key", "entry_type": "article", "title": "Paper A", "year": 2024}]

    def fake_dedup(raw_entries: list[dict], arxiv_id: str = ""):
        return (
            [{
                "arxiv_id": arxiv_id,
                "citation_key": "new_key",
                "entry_type": "article",
                "title": "Paper A",
                "authors": ["Alice"],
                "year": 2024,
                "doi": None,
                "journal": None,
                "url": None,
            }],
            {"old_key": "new_key", "new_key": "new_key"},
        )

    def fake_rewrite(compiled_tex: str, key_map: dict[str, str]) -> str:
        out = compiled_tex
        for old, new in key_map.items():
            out = out.replace(f"\\cite{{{old}}}", f"\\cite{{{new}}}")
        return out

    run_mod._resolve_bibtex_hooks = lambda: (fake_extract, fake_dedup, fake_rewrite)

    out_dir = tmp_path / "processed"
    try:
        processed = run_parser(
            base_ids=["2607.23694", "2607.23695"],
            source_dir=source_dir,
            source_manifest=source_manifest,
            output_dir=out_dir,
            follow=False,
        )
    finally:
        run_mod.DISCOVERY_FILE = orig_discovery
        run_mod.RAW_DIR = orig_raw

    assert processed == 2
    bib_table = pq.read_table(out_dir / "bibtex.parquet")
    assert pa.types.is_int32(bib_table.schema.field("year").type)
    assert pa.types.is_list(bib_table.schema.field("authors").type)

    with (out_dir / "parse_manifest.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["arxiv_id"], r["version"]): r for r in rows}
    assert by_key[("2607.23694", "1")]["status"] == "parsed"
    assert by_key[("2607.23694", "1")]["bibtex_raw_count"] == "1"
    assert by_key[("2607.23695", "1")]["status"] == "empty"


def test_repeated_sentence_in_different_sections_has_distinct_parents():
    builder = _parse_document(
        r"\section{First}The same sentence. "
        r"\section{Second}The same sentence."
    )

    sentences = [
        row for row in builder.elements_rows
        if row["kind"] == "sentence" and row["content"] == "The same sentence."
    ]
    assert len(sentences) == 2
    assert sentences[0]["element_id"] != sentences[1]["element_id"]

    sentence_parents = {
        row["element_id"]: row["parent_id"]
        for row in builder.hierarchy_rows
        if row["element_id"] in {sentence["element_id"] for sentence in sentences}
    }
    assert len(set(sentence_parents.values())) == 2
    _assert_single_parent_per_version(builder)


def test_duplicate_list_items_have_distinct_ids():
    builder = _parse_document(
        r"\section{Results}"
        r"\begin{itemize}\item Same item.\item Same item.\end{itemize}"
    )

    items = [
        row for row in builder.elements_rows
        if row["kind"] == "item" and row["content"] == "Same item."
    ]
    assert len(items) == 2
    assert items[0]["element_id"] != items[1]["element_id"]
    _assert_single_parent_per_version(builder)


def test_repeated_heading_title_does_not_create_self_parent():
    builder = _parse_document(
        r"\section{Results}Section text. "
        r"\subsection{Results}Subsection text."
    )

    results_headings = [
        row for row in builder.elements_rows
        if row["kind"] == "heading" and row["content"] == "Results"
    ]
    assert len(results_headings) == 2
    assert results_headings[0]["element_id"] != results_headings[1]["element_id"]

    parents = {
        row["element_id"]: row["parent_id"]
        for row in builder.hierarchy_rows
    }
    assert parents[results_headings[1]["element_id"]] == results_headings[0]["element_id"]
    assert all(element_id != parent_id for element_id, parent_id in parents.items())
    _assert_single_parent_per_version(builder)


def test_nested_itemize_preserves_nested_hierarchy():
    builder = _parse_document(
        r"\section{Results}"
        r"\begin{itemize}"
        r"\item Outer item. "
        r"\begin{itemize}"
        r"\item Inner one.\item Inner two."
        r"\end{itemize}"
        r"\item Outer second."
        r"\end{itemize}"
    )

    lists = [row for row in builder.elements_rows if row["kind"] == "itemize"]
    items = [row for row in builder.elements_rows if row["kind"] == "item"]
    assert len(lists) == 2
    assert len(items) == 4

    parent_by_element = {
        row["element_id"]: row["parent_id"]
        for row in builder.hierarchy_rows
    }
    nested_list = next(
        row for row in lists
        if parent_by_element[row["element_id"]] in {item["element_id"] for item in items}
    )
    assert len(
        [
            item for item in items
            if parent_by_element[item["element_id"]] == nested_list["element_id"]
        ]
    ) == 2
    _assert_single_parent_per_version(builder)
