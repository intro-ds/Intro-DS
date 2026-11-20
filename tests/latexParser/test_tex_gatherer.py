"""Unit tests for the tex_gatherer module."""

from pathlib import Path
import pytest

from src.latexParser.tex_gatherer import find_root_tex, gather_tex_files


def test_find_root_tex_by_documentclass(tmp_path: Path) -> None:
    """Test finding the root tex file containing \\documentclass."""
    sub_tex = tmp_path / "section1.tex"
    sub_tex.write_text("Just section content", encoding="utf-8")

    main_tex = tmp_path / "main.tex"
    main_tex.write_text(
        "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}",
        encoding="utf-8",
    )

    root = find_root_tex(tmp_path)
    assert root is not None
    assert root.name == "main.tex"


def test_find_root_tex_ignores_commented(tmp_path: Path) -> None:
    """Test that commented out \\documentclass is ignored."""
    file1 = tmp_path / "draft.tex"
    file1.write_text(
        "% \\documentclass{article}\n% Old commented template\nContent",
        encoding="utf-8",
    )

    file2 = tmp_path / "real.tex"
    file2.write_text(
        "\\documentclass[12pt]{report}\n\\begin{document}\nReal\n\\end{document}",
        encoding="utf-8",
    )

    root = find_root_tex(tmp_path)
    assert root is not None
    assert root.name == "real.tex"


def test_find_root_tex_none_when_empty(tmp_path: Path) -> None:
    """Test that None is returned when no tex files exist."""
    assert find_root_tex(tmp_path) is None


def test_gather_tex_files_follows_input(tmp_path: Path) -> None:
    """Test that \\input and \\include are followed in traversal order."""
    root = tmp_path / "main.tex"
    root.write_text(
        "\\documentclass{article}\n\\input{intro}\n\\include{methods}\n\\end{document}",
        encoding="utf-8",
    )

    intro = tmp_path / "intro.tex"
    intro.write_text("Introduction prose here.", encoding="utf-8")

    methods = tmp_path / "methods.tex"
    methods.write_text("Methods prose here.\n\\input{algo}", encoding="utf-8")

    algo = tmp_path / "algo.tex"
    algo.write_text("Algorithm details.", encoding="utf-8")

    unused = tmp_path / "unused.tex"
    unused.write_text("This should not be included.", encoding="utf-8")

    gathered = gather_tex_files(root, tmp_path)
    gathered_names = [p.name for p in gathered]

    assert gathered_names == ["main.tex", "intro.tex", "methods.tex", "algo.tex"]
    assert "unused.tex" not in gathered_names


def test_gather_tex_files_circular_protection(tmp_path: Path) -> None:
    """Test circular dependency protection during transitive gathering."""
    file_a = tmp_path / "a.tex"
    file_b = tmp_path / "b.tex"

    file_a.write_text("\\documentclass{article}\n\\input{b}", encoding="utf-8")
    file_b.write_text("\\input{a}\nContent B", encoding="utf-8")

    gathered = gather_tex_files(file_a, tmp_path)
    names = [p.name for p in gathered]
    assert names == ["a.tex", "b.tex"]


def test_find_root_tex_ignores_pdf_or_binary(tmp_path: Path) -> None:
    """Test that binary PDF files disguised as .tex are ignored."""
    pdf_tex = tmp_path / "main.tex"
    pdf_tex.write_bytes(b"%PDF-1.7\n\x00\x01\x02\\documentclass{article}")

    assert find_root_tex(tmp_path) is None

