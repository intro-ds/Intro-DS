from pathlib import Path

from src.latexParser.file_resolver import LatexFileResolver

DOC = "\\documentclass{article}\n\\begin{document}\nBODY\n\\end{document}"


def test_root_prefers_file_matching_bbl(tmp_path: Path) -> None:
    (tmp_path / "elsdoc.tex").write_text(DOC.replace("BODY", "Template"), encoding="utf-8")
    (tmp_path / "paper_v2.tex").write_text(DOC.replace("BODY", "Real " * 50), encoding="utf-8")
    (tmp_path / "paper_v2_highlight.tex").write_text(DOC.replace("BODY", "Real " * 50), encoding="utf-8")
    (tmp_path / "paper_v2_highlight.bbl").write_text("\\begin{thebibliography}{1}", encoding="utf-8")
    assert LatexFileResolver().find_root_tex_file(tmp_path).name == "paper_v2_highlight.tex"


def test_root_skips_file_included_by_another_candidate(tmp_path: Path) -> None:
    (tmp_path / "a.tex").write_text(DOC.replace("BODY", "\\input{b}"), encoding="utf-8")
    (tmp_path / "b.tex").write_text(DOC.replace("BODY", "x " * 200), encoding="utf-8")
    assert LatexFileResolver().find_root_tex_file(tmp_path).name == "a.tex"


def test_root_ignores_pdf_disguised_as_tex(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_bytes(b"%PDF-1.7\n\x00\x01\\documentclass{article}")
    assert LatexFileResolver().find_root_tex_file(tmp_path) is None


def test_input_resolves_by_basename_when_folders_were_flattened(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text(DOC.replace("BODY", "\\input{sections/intro}"), encoding="utf-8")
    (tmp_path / "intro.tex").write_text("Intro text.", encoding="utf-8")
    res = LatexFileResolver().resolve_version(tmp_path)
    assert "Intro text." in res["compiled_tex"]
    assert res["ignored_files"] == []


def test_includegraphics_is_not_treated_as_include(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text(
        DOC.replace("BODY", "\\includegraphics[width=\\linewidth]{fig.png}"), encoding="utf-8"
    )
    res = LatexFileResolver().resolve_version(tmp_path)
    assert "\\includegraphics[width=\\linewidth]{fig.png}" in res["compiled_tex"]