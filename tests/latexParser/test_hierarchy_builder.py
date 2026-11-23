"""Unit tests for the hierarchy_builder module."""

from pathlib import Path
import pytest

from src.latexParser.hierarchy_builder import (
    KIND_DOCUMENT,
    KIND_FIGURE,
    KIND_FORMULA,
    KIND_HEADING,
    KIND_ITEM,
    KIND_ITEMIZE,
    KIND_SENTENCE,
    build_hierarchy,
    flatten_tree,
    tree_to_hierarchy_rows,
)


def test_build_hierarchy_simple_sections(tmp_path: Path) -> None:
    """Test building hierarchy from document with sections and prose."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \title{Sample Paper Title}
        \section{Introduction}
        First sentence of introduction. Second sentence of introduction.
        \section{Methods}
        Methods sentence.
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00001", 1)
    assert root.kind == KIND_DOCUMENT
    assert root.element_id.startswith("2512-00001#")

    all_elems = flatten_tree(root)
    kinds = [e.kind for e in all_elems]
    assert KIND_HEADING in kinds
    assert KIND_SENTENCE in kinds

    headings = [e.content for e in all_elems if e.kind == KIND_HEADING]
    assert "Introduction" in headings
    assert "Methods" in headings


def test_build_hierarchy_excludes_references(tmp_path: Path) -> None:
    """Test that References section is excluded from the hierarchy."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Introduction}
        Intro content.
        \section{References}
        \bibitem{a} Ref A
        \bibitem{b} Ref B
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00002", 1)
    all_elems = flatten_tree(root)
    headings = [e.content for e in all_elems if e.kind == KIND_HEADING]
    assert "Introduction" in headings
    assert "References" not in headings


def test_build_hierarchy_includes_star_sections(tmp_path: Path) -> None:
    """Test that starred sections like Acknowledgements are included."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Introduction}
        Intro.
        \section*{Acknowledgements}
        We thank the funding agencies.
        \section*{Appendix}
        Extra derivations here.
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00003", 1)
    all_elems = flatten_tree(root)
    headings = [e.content for e in all_elems if e.kind == KIND_HEADING]
    assert "Acknowledgements" in headings
    assert "Appendix" in headings


def test_build_hierarchy_with_formula_and_figure_and_itemize(tmp_path: Path) -> None:
    """Test that block formulas, figures, and lists are parsed properly."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Experiments}
        We test our formulation:
        \begin{equation}
        y = f(x) + \epsilon
        \end{equation}
        \begin{figure}[ht]
        \caption{Overview of our model pipeline.}
        \end{figure}
        \begin{itemize}
        \item First observation.
        \item Second observation.
        \end{itemize}
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00004", 1)
    all_elems = flatten_tree(root)
    kinds = [e.kind for e in all_elems]

    assert KIND_FORMULA in kinds
    assert KIND_FIGURE in kinds
    assert KIND_ITEMIZE in kinds
    assert KIND_ITEM in kinds

    figures = [e for e in all_elems if e.kind == KIND_FIGURE]
    assert "Overview of our model pipeline" in figures[0].content


def test_tree_to_hierarchy_rows(tmp_path: Path) -> None:
    """Test converting a tree to hierarchy edge rows."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Overview}
        Single sentence here.
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00005", 1)
    rows = tree_to_hierarchy_rows(root, "2512.00005", 1)
    assert len(rows) >= 2
    for r in rows:
        assert r["arxiv_id"] == "2512.00005"
        assert r["version"] == 1
        assert "element_id" in r
        assert "parent_id" in r

def test_build_hierarchy_nests_subsections(tmp_path: Path) -> None:
    """Test that subsections are nested under their parent section."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Methods}
        Methods intro sentence.
        \subsection{Data}
        Data sentence.
        \subsection{Model}
        Model sentence.
        \section{Results}
        Results sentence.
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00006", 1)
    top = [c for c in root.children if c.kind == KIND_HEADING]
    assert [h.content for h in top] == ["Methods", "Results"]

    subs = [c for c in top[0].children if c.kind == KIND_HEADING]
    assert [s.content for s in subs] == ["Data", "Model"]

def test_build_hierarchy_keeps_appendix_after_references(tmp_path: Path) -> None:
    """Test that appendices placed after the bibliography are still parsed."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Introduction}
        Intro.
        \begin{thebibliography}{9}
        \bibitem{a} Ref A.
        \end{thebibliography}
        \appendix
        \section{Proof}
        Proof sentence.
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00007", 1)
    all_elems = flatten_tree(root)
    headings = [e.content for e in all_elems if e.kind == KIND_HEADING]
    assert "Proof" in headings
    assert not any("Ref A" in e.content for e in all_elems)


def test_build_hierarchy_drops_references_section_keeps_next(tmp_path: Path) -> None:
    """Test that a References section is dropped but a following appendix is kept."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Introduction}
        Intro.
        \section{References}
        \bibitem{a} Ref A
        \appendix
        \section{Extra}
        Extra sentence.
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00008", 1)
    headings = [e.content for e in flatten_tree(root) if e.kind == KIND_HEADING]
    assert "References" not in headings
    assert "Extra" in headings

def test_build_hierarchy_ids_unique_across_versions(tmp_path: Path) -> None:
    """Test that element IDs never collide between versions of one paper."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Intro}
        A sentence.
        \end{document}
        """,
        encoding="utf-8",
    )

    root_v1 = build_hierarchy([tex_file], "2512.00009", 1)
    root_v2 = build_hierarchy([tex_file], "2512.00009", 2)
    ids_v1 = {e.element_id for e in flatten_tree(root_v1)}
    ids_v2 = {e.element_id for e in flatten_tree(root_v2)}
    assert ids_v1.isdisjoint(ids_v2)
    assert root_v1.content != root_v2.content

def test_build_hierarchy_expands_input_files(tmp_path: Path) -> None:
    """Test that content of \\input files is parsed in the document body."""
    main = tmp_path / "main.tex"
    main.write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{intro}\n\\end{document}",
        encoding="utf-8",
    )
    intro = tmp_path / "intro.tex"
    intro.write_text("\\section{Introduction}\nIntro sentence here.", encoding="utf-8")

    root = build_hierarchy([main, intro], "2512.00010", 1)
    all_elems = flatten_tree(root)
    headings = [e.content for e in all_elems if e.kind == KIND_HEADING]
    sentences = [e.content for e in all_elems if e.kind == KIND_SENTENCE]
    assert "Introduction" in headings
    assert any("Intro sentence here" in s for s in sentences)
    assert not any("intro.tex" in s for s in sentences)


def test_build_hierarchy_ignores_definitions_in_body(tmp_path: Path) -> None:
    """Test that \\newcommand and similar definitions do not become sentences."""
    tex_file = tmp_path / "paper.tex"
    tex_file.write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \newcommand{\ours}{OurMethod}
        \DeclareMathOperator{\argmax}{arg\,max}
        \section{Intro}
        Real sentence.
        \end{document}
        """,
        encoding="utf-8",
    )

    root = build_hierarchy([tex_file], "2512.00011", 1)
    contents = " ".join(e.content for e in flatten_tree(root))
    assert "OurMethod" not in contents
    assert "argmax" not in contents
    assert "Real sentence" in contents