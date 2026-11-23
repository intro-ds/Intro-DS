"""Unit tests for the standardizer module."""

from pathlib import Path
import pytest

from src.latexParser.standardizer import (
    bibitem_to_bibtex,
    clean_element_content,
    expand_accents,
    extract_bibitems_from_tex,
    extract_citation_keys,
    normalize_block_math,
    normalize_inline_math,
    parse_bib_file,
    split_sentences,
)


def test_clean_formatting_commands() -> None:
    """Test removing formatting markup like \\textbf, \\emph, \\centering."""
    raw = r"\centering \textbf{Bold text} and \emph{emphasized} text with \midrule."
    cleaned = clean_element_content(raw, kind="sentence")
    assert "Bold text" in cleaned
    assert "emphasized" in cleaned
    assert r"\textbf" not in cleaned
    assert r"\centering" not in cleaned
    assert r"\midrule" not in cleaned


def test_normalize_inline_math() -> None:
    """Test converting \\(...) and \\begin{math} to $...$."""
    raw = r"Here is \(x + y = z\) and \begin{math}\alpha\end{math}."
    norm = normalize_inline_math(raw)
    assert "$x + y = z$" in norm
    assert r"$\alpha$" in norm


def test_normalize_block_math() -> None:
    """Test normalizing display math, align, and $$ to \\begin{equation}."""
    raw = r"$$E = mc^2$$"
    norm = normalize_block_math(raw)
    assert norm == r"\begin{equation}E = mc^2\end{equation}"

    raw_bracket = r"\[a^2 + b^2 = c^2\]"
    norm_bracket = normalize_block_math(raw_bracket)
    assert norm_bracket == r"\begin{equation}a^2 + b^2 = c^2\end{equation}"


def test_expand_accents() -> None:
    """Test expanding LaTeX accent commands to Unicode."""
    raw = r"Schr\"odinger and Poincar\'e went to Z\"urich."
    expanded = expand_accents(raw)
    assert "Schrödinger" in expanded or "Schr\u00f6dinger" in expanded
    assert "Poincaré" in expanded or "Poincar\u00e9" in expanded
    assert "Zürich" in expanded or "Z\u00fcrich" in expanded


def test_split_sentences_basic() -> None:
    """Test simple sentence splitting."""
    text = "This is the first sentence. This is the second sentence. And the third."
    s = split_sentences(text)
    assert len(s) == 3
    assert s[0] == "This is the first sentence"
    assert s[1] == "This is the second sentence"
    assert s[2] == "And the third"


def test_split_sentences_no_split_on_abbrev_and_math() -> None:
    """Test that splitting does not occur on abbreviations, math, or decimals."""
    text = "We refer to Fig. 1 and Eq. 2 for the model. The value is $x = 3.14$ as shown by Smith et al. in 2020."
    s = split_sentences(text)
    assert len(s) == 2
    assert "Fig. 1" in s[0]
    assert "Eq. 2" in s[0]
    assert "$x = 3.14$" in s[1]
    assert "et al." in s[1]


def test_bibitem_to_bibtex() -> None:
    """Test converting \\bibitem to structured BibTeX."""
    raw = r"A. Vaswani, N. Shazeer, and N. Parmar, ``Attention Is All You Need'', 2017. doi:10.1234/test"
    entry = bibitem_to_bibtex("vaswani2017", raw)

    assert entry["citation_key"] == "vaswani2017"
    assert entry["year"] == 2017
    assert entry["doi"] == "10.1234/test"
    assert entry["title"] is not None
    assert "Attention Is All You Need" in entry["title"]
    assert len(entry["authors"]) >= 1


def test_parse_bib_file(tmp_path: Path) -> None:
    """Test parsing standard BibTeX .bib file."""
    bib_content = """
    @article{einstein1905,
        author = {Albert Einstein and Max Planck},
        title = {On the electrodynamics of moving bodies},
        journal = {Annalen der Physik},
        year = {1905},
        doi = {10.1002/andp.19053221004}
    }
    """
    bib_file = tmp_path / "refs.bib"
    bib_file.write_text(bib_content, encoding="utf-8")

    entries = parse_bib_file(bib_file)
    assert len(entries) == 1
    e = entries[0]
    assert e["citation_key"] == "einstein1905"
    assert e["entry_type"] == "article"
    assert e["year"] == 1905
    assert "Albert Einstein" in e["authors"]
    assert "electrodynamics" in e["title"]


def test_extract_citation_keys() -> None:
    """Test extracting citation keys from LaTeX text."""
    tex = r"We cite \cite{einstein1905, bohr1913} and also \citep[see][p. 10]{feynman1965}."
    keys = extract_citation_keys(tex)
    assert set(keys) == {"einstein1905", "bohr1913", "feynman1965"}


def test_parse_bib_file_large_filtering(tmp_path: Path) -> None:
    """Test that large bib files filter to only cited keys."""
    # Generate 550 dummy entries
    lines = []
    for i in range(550):
        lines.append(f"@article{{key{i},\n  title = {{Paper {i}}},\n  author = {{Author {i}}}\n}}")
    bib_file = tmp_path / "large.bib"
    bib_file.write_text("\n".join(lines), encoding="utf-8")

    # Only cite key5 and key42
    entries = parse_bib_file(bib_file, filter_keys={"key5", "key42"})
    assert len(entries) == 2
    keys = {e["citation_key"] for e in entries}
    assert keys == {"key5", "key42"}

def test_expand_accents_keeps_long_commands() -> None:
    """Test that \\label, \\linewidth, \\LaTeX, \\orcidlink are not corrupted."""
    raw = r"\label{sec:a} 0.9\linewidth \LaTeX{} \orcidlink{0000} Gro\ss{} \o{}"
    out = expand_accents(raw)
    assert r"\label{sec:a}" in out
    assert r"\linewidth" in out
    assert r"\LaTeX" in out
    assert r"\orcidlink" in out
    assert "Groß" in out
    assert "ø" in out

    cleaned = clean_element_content(r"Intro \label{sec:a} text.", kind="sentence")
    assert "label" not in cleaned
    assert "ł" not in cleaned

def test_extract_bibitems_revtex_bbl() -> None:
    """Test parsing revtex-style .bbl entries with a space before [label]."""
    bbl = r"""
    \begin{thebibliography}{2}
    \bibitem [{\citenamefont {Foo}\ and\ \citenamefont {Bar}(2020)}]{foo2020}%
      \BibitemOpen
      \bibfield  {author} {\bibinfo {author} {\bibfnamefont {A.}~\bibnamefont {Foo}}\ and\ \bibinfo {author} {\bibfnamefont {B.}\ \bibfnamefont {C.}~\bibnamefont {Bar}},\ }\bibfield  {title} {\emph {\bibinfo {title} {Gravity channels}},\ }\href {x} {\bibfield  {journal} {\bibinfo  {journal} {Phys. Rev. Lett.}\ }\textbf {\bibinfo {volume} {1}},\ \bibinfo {pages} {1} (\bibinfo {year} {2020})}\BibitemShut {NoStop}%
    \bibitem{plain1} A. Smith, Title, 2019.
    \end{thebibliography}
    """
    entries = extract_bibitems_from_tex(bbl)
    assert [e["citation_key"] for e in entries] == ["foo2020", "plain1"]
    first = entries[0]
    assert first["title"] == "Gravity channels"
    assert first["year"] == 2020
    assert first["journal"] == "Phys. Rev. Lett."
    assert first["authors"] == ["A. Foo", "B. C. Bar"]