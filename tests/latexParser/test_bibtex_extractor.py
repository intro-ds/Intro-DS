from pathlib import Path

from src.latexParser.bibtex_extractor import extract_bibtex


def test_extract_bibtex_collects_bibitem_and_bib_file(tmp_path: Path) -> None:
    (tmp_path / "refs.bib").write_text(
        "@article{einstein1905,\n title={On the electrodynamics of moving bodies},\n"
        " author={Albert Einstein},\n year={1905}\n}",
        encoding="utf-8",
    )
    tex = (
        "\\begin{thebibliography}{1}\n"
        "\\bibitem{smith2019} J. Smith. Title of paper. Journal of Things, 2019.\n"
        "\\end{thebibliography}"
    )
    entries = extract_bibtex(tex, tmp_path)
    assert {e["citation_key"] for e in entries} == {"smith2019", "einstein1905"}


def test_extract_bibtex_falls_back_to_bbl(tmp_path: Path) -> None:
    (tmp_path / "paper.bbl").write_text(
        "\\begin{thebibliography}{1}\n\\bibitem{a} A. Foo. Title here. 2020.\n"
        "\\end{thebibliography}",
        encoding="utf-8",
    )
    entries = extract_bibtex("\\documentclass{article}", tmp_path)
    assert [e["citation_key"] for e in entries] == ["a"]