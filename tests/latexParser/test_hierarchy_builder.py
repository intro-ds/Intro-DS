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
