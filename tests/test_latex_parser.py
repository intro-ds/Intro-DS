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


def test_multi_file_and_deduplication(tmp_path: Path):
    # Giả lập bài báo 2310-12345 với 2 version như trong đặc tả parsing.md
    v1_dir = tmp_path / "2310-12345" / "v1"
    v2_dir = tmp_path / "2310-12345" / "v2"
    v1_dir.mkdir(parents=True)
    v2_dir.mkdir(parents=True)

    # Version 1: tách làm main.tex + intro.tex + unused.tex (file thừa)
    (v1_dir / "main.tex").write_text(
        r"""
        \documentclass{article}
        \begin{document}
        % \input{unused.tex}
        \input{intro}
        \section*{References}
        \bibitem{ref1} Ignored reference section.
        \end{document}
        """,
        encoding="utf-8",
    )
    (v1_dir / "intro.tex").write_text(
        r"""
        \section{Introduction}
        \centering We study the citation matching problem.
        """,
        encoding="utf-8",
    )
    (v1_dir / "unused.tex").write_text(
        r"\section{Should Not Be Parsed} Secret text.", encoding="utf-8"
    )

    (v2_dir / "main.tex").write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Background and model}
        We study the citation matching problem.
        \end{document}
        """,
        encoding="utf-8",
    )

    resolver = LatexFileResolver()
    builder = PublicationHierarchyBuilder("2310.12345")

    res_v1 = resolver.resolve_version(v1_dir)
    assert not res_v1["is_empty"]
    assert "unused.tex" in res_v1["ignored_files"]
    builder.parse_version(1, res_v1["compiled_tex"])

    res_v2 = resolver.resolve_version(v2_dir)
    builder.parse_version(2, res_v2["compiled_tex"])

    sentences = [e for e in builder.elements_rows if e["kind"] == "sentence"]
    assert len(sentences) == 1
    assert sentences[0]["element_id"] == "2310-12345#s1"

    headings = [e["content"] for e in builder.elements_rows if e["kind"] == "heading"]
    assert "References" not in headings
    assert headings == ["Introduction", "Background and model"]
    _assert_single_parent_per_version(builder)


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