import logging
from pathlib import Path
from src.latexParser import LatexFileResolver, PublicationHierarchyBuilder

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

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

def test_root_detection_with_multiple_documentclass(tmp_path: Path):
    _write(tmp_path / "main.tex",
           "\\documentclass{article}\n\\begin{document}\n\\input{intro}\n\\end{document}")
    _write(tmp_path / "intro.tex", "Intro text.")
    # extra files that also look like roots, or only mention \documentclass in a comment
    _write(tmp_path / "rebuttal.tex",
           "\\documentclass{letter}\n\\begin{document}Reply\\end{document}")
    _write(tmp_path / "supp" / "supplement.tex",
           "\\documentclass{article}\n\\begin{document}" + "Long supplement. " * 200
           + "\\end{document}")
    _write(tmp_path / "old.tex", "% \\documentclass{article}\nSome notes")

    res = LatexFileResolver().resolve_version(tmp_path)

    assert not res["is_empty"]
    assert res["root_file"] == "main.tex"
    assert [Path(p).as_posix() for p in res["included_files"]] == ["main.tex", "intro.tex"]
    assert {Path(p).as_posix() for p in res["ignored_files"]} == {
        "rebuttal.tex", "supp/supplement.tex", "old.tex"}


def test_version_without_tex_is_empty(tmp_path: Path):
    no_files = tmp_path / "v1"
    no_files.mkdir()
    bib_only = tmp_path / "v2"
    _write(bib_only / "refs.bib", "@article{x, title={T}}")
    no_root = tmp_path / "v3"
    _write(no_root / "notes.tex", "Just notes, no preamble.")

    resolver = LatexFileResolver()
    for version_dir in (no_files, bib_only):
        res = resolver.resolve_version(version_dir)
        assert res["is_empty"] and res["root_file"] is None
        assert res["compiled_tex"] == "" and res["missing_files"] == []

    res = resolver.resolve_version(no_root)   # .tex exists but no documentclass/begin{document}
    assert res["is_empty"]
    assert [Path(p).as_posix() for p in res["ignored_files"]] == ["notes.tex"]

    # an empty version yields no parsed elements
    builder = PublicationHierarchyBuilder("2310.12345")
    assert builder.parse_version(1, res["compiled_tex"]) is None
    assert builder.elements_rows == []


def test_input_cycle_terminates(tmp_path: Path):
    _write(tmp_path / "main.tex",
           "\\documentclass{article}\n\\begin{document}\nA1 \\input{b}\n\\end{document}")
    _write(tmp_path / "b.tex", "B1 \\input{main}")   # b includes the root again

    res = LatexFileResolver().resolve_version(tmp_path)

    assert [Path(p).as_posix() for p in res["included_files"]] == ["main.tex", "b.tex"]
    assert "A1" in res["compiled_tex"] and "B1" in res["compiled_tex"]
    assert res["compiled_tex"].count("\\documentclass") == 1   # root not expanded twice
    assert res["missing_files"] == []


def test_missing_include_is_reported(tmp_path: Path, caplog):
    _write(tmp_path / "main.tex",
           "\\documentclass{article}\n\\begin{document}\n"
           "Before. \\input{chapters/missing}\n"
           "\\includegraphics[width=\\linewidth]{fig}\n"
           "After.\n\\end{document}")

    with caplog.at_level(logging.WARNING, logger="src.latexParser.file_resolver"):
        res = LatexFileResolver().resolve_version(tmp_path)

    assert not res["is_empty"]
    assert res["missing_files"] == ["chapters/missing"]      # partial version is detectable
    assert res["included_files"] == ["main.tex"]
    assert any("chapters/missing" in r.getMessage() for r in caplog.records)
    # the rest of the text is kept, and \includegraphics is NOT treated as an include
    assert "Before." in res["compiled_tex"] and "After." in res["compiled_tex"]
    assert "\\includegraphics[width=\\linewidth]{fig}" in res["compiled_tex"]


def test_commented_input_ignored_and_escaped_percent_is_not_comment(tmp_path: Path):
    _write(tmp_path / "main.tex",
           "\\documentclass{article}\n\\begin{document}\n"
           "% \\input{ghost}\n"
           "Growth of 50\\% \\input{part2}\n"
           "\\input{part3} % \\input{ghost2}\n"
           "\\end{document}")
    _write(tmp_path / "part2.tex", "PART TWO TEXT")
    _write(tmp_path / "part3.tex", "PART THREE TEXT")

    res = LatexFileResolver().resolve_version(tmp_path)

    # \% is not a comment, so the \input after it is still followed
    assert "PART TWO TEXT" in res["compiled_tex"]
    assert "50\\%" in res["compiled_tex"]
    # real comments are plain text: not followed, and not reported as missing
    assert "PART THREE TEXT" in res["compiled_tex"]
    assert [Path(p).as_posix() for p in res["included_files"]] == [
        "main.tex", "part2.tex", "part3.tex"]
    assert res["missing_files"] == []
    assert "ghost" not in res["compiled_tex"]