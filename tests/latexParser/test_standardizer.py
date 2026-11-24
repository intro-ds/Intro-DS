"""Unit tests for the standardizer module."""

from pathlib import Path
import pytest
import re

from src.latexParser.standardizer import (
    _split_authors,
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

def test_bibitem_newblock_natbib_style() -> None:
    """Test natbib/plain style entries separated by \\newblock."""
    raw = (
        "Ashish Vaswani, Noam Shazeer, and Niki Parmar.\n"
        "\\newblock Attention is all you need.\n"
        "\\newblock In \\emph{Advances in Neural Information Processing Systems}, "
        "volume 30, 2017."
    )
    e = bibitem_to_bibtex("vaswani", raw)
    assert e["title"] == "Attention is all you need"
    assert e["authors"] == ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
    assert e["year"] == 2017
    assert e["journal"] == "Advances in Neural Information Processing Systems"
    assert e["entry_type"] == "inproceedings"


def test_bibitem_last_first_apa_style() -> None:
    """Test "Last, F., Last, G., and Last, H. (year). Title. Venue." entries."""
    raw = "Vaswani, A., Shazeer, N., and Parmar, N. (2017). Attention is all you need. \\emph{NeurIPS}."
    e = bibitem_to_bibtex("vaswani", raw)
    assert e["authors"][0] == "A. Vaswani"
    assert len(e["authors"]) == 3
    assert e["year"] == 2017
    assert e["title"] == "Attention is all you need"


def test_bibitem_arxiv_id_becomes_url() -> None:
    """Test that an arXiv identifier is kept as url and the year is not confused."""
    raw = "K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning. arXiv preprint arXiv:1512.03385, 2015."
    e = bibitem_to_bibtex("he", raw)
    assert e["url"] == "https://arxiv.org/abs/1512.03385"
    assert e["year"] == 2015
    assert e["entry_type"] == "misc"


def test_split_authors_formats() -> None:
    """Test the two common author-list formats."""
    assert _split_authors("A. B. Foo, C. Bar, and D. Baz") == ["A. B. Foo", "C. Bar", "D. Baz"]
    assert _split_authors("Foo, A. B. and Bar, C.") == ["A. B. Foo", "C. Bar"]

def test_bibitem_plain_style_doi_and_year() -> None:
    """Test a plain-style entry without \\newblock: DOI and year are still found."""
    raw = "J. Smith. Title of paper. Journal of Things, 12(3):45--67, 2019. doi:10.1000/xyz123"
    e = bibitem_to_bibtex("smith2019", raw)
    assert e["doi"] == "10.1000/xyz123"
    assert e["year"] == 2019
    assert e["entry_type"] == "article"


def test_bibitem_url_command() -> None:
    """Test that a \\url{...} link is kept and not mistaken for the year."""
    raw = (
        "D. Kingma and J. Ba. Adam: A method for stochastic optimization. "
        "\\url{https://arxiv.org/abs/1412.6980}, 2014."
    )
    e = bibitem_to_bibtex("adam", raw)
    assert e["url"] == "https://arxiv.org/abs/1412.6980"
    assert e["year"] == 2014


def test_extract_bibitems_with_optional_label() -> None:
    """Test \\bibitem[label]{key} with \\newblock-separated fields."""
    tex = (
        "\\begin{thebibliography}{1}\n"
        "\\bibitem[Smith et al.(2020)]{smith2020} J. Smith.\n"
        "\\newblock Great title.\n"
        "\\newblock Nature, 2020.\n"
        "\\end{thebibliography}"
    )
    entries = extract_bibitems_from_tex(tex)
    assert len(entries) == 1
    e = entries[0]
    assert e["citation_key"] == "smith2020"
    assert e["title"] == "Great title"
    assert e["authors"] == ["J. Smith"]
    assert e["year"] == 2020
    assert e["journal"] == "Nature"


def test_bibitem_without_structure_does_not_crash() -> None:
    """Test that an unstructured entry yields empty fields instead of an error."""
    e = bibitem_to_bibtex("web1", "Some website.")
    assert e["title"] is None
    assert e["year"] is None
    assert e["authors"] == []
    assert e["entry_type"] == "misc"


def test_parse_bib_file_inproceedings_and_misc(tmp_path: Path) -> None:
    """Test .bib with quoted values, nested braces, bare year, booktitle and url."""
    bib = tmp_path / "refs.bib"
    bib.write_text(
        """
@inproceedings{vaswani2017,
  author = "Ashish Vaswani and Noam Shazeer",
  title = {Attention Is {All} You Need},
  booktitle = {Advances in Neural Information Processing Systems},
  year = 2017
}
@misc{adam,
  author = {Diederik Kingma and Jimmy Ba},
  title = {Adam},
  year = {2014},
  url = {https://arxiv.org/abs/1412.6980}
}
""",
        encoding="utf-8",
    )
    entries = parse_bib_file(bib)
    assert [e["citation_key"] for e in entries] == ["vaswani2017", "adam"]

    first, second = entries
    assert first["entry_type"] == "inproceedings"
    assert first["title"] == "Attention Is All You Need"
    assert first["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert first["year"] == 2017
    assert first["journal"] == "Advances in Neural Information Processing Systems"

    assert second["entry_type"] == "misc"
    assert second["journal"] is None
    assert second["url"] == "https://arxiv.org/abs/1412.6980"


def test_extract_bibitems_real_bbl_fixture() -> None:
    """Test a real .bbl excerpt: one entry per \\bibitem and most have a title."""
    fixture = Path(__file__).parent / "fixtures" / "sample.bbl"
    text = fixture.read_text(encoding="utf-8", errors="replace")
    entries = extract_bibitems_from_tex(text)
    assert len(entries) == len(re.findall(r"\\bibitem\b", text))
    assert len(entries) >= 5
    assert all(e["citation_key"] for e in entries)
    assert sum(bool(e["authors"]) and bool(e["year"]) for e in entries) / len(entries) >= 0.8

def test_parse_bib_file_skips_control_entries(tmp_path: Path) -> None:
    """Test that REVTeX @CONTROL directives are not parsed as references."""
    bib = tmp_path / "refs.bib"
    bib.write_text(
        """
@CONTROL{REVTEX42Control}
@CONTROL{apsrev42Control,
  author="08", editor="1", pages="0", title="0", year="1"}
@article{einstein1905,
  author = {Albert Einstein},
  title = {On the electrodynamics of moving bodies},
  year = {1905}
}
""",
        encoding="utf-8",
    )
    entries = parse_bib_file(bib)
    assert [e["citation_key"] for e in entries] == ["einstein1905"]

def test_revtex_title_with_nested_braces_and_textendash() -> None:
    """Test a revtex title wrapped in {{...}} with \\textendash{} and a line break."""
    tex = (
        "\\begin{thebibliography}{1}\n"
        "\\bibitem [{\\citenamefont {Roulet}\\ and\\ \\citenamefont {Zaldarriaga}(2019)}]{Roulet:2018jbe}%\n"
        "  \\BibitemOpen\n"
        "  \\bibfield  {author} {\\bibinfo {author} {\\bibfnamefont {J.}~\\bibnamefont {Roulet}}\\ and\\ "
        "\\bibinfo {author} {\\bibfnamefont {M.}~\\bibnamefont {Zaldarriaga}},\\ }"
        "\\bibfield  {title} {\\bibinfo {title} {{Constraints on\n"
        "  binary black hole populations from LIGO\\textendash{}Virgo detections}},\\ }"
        "\\href {https://doi.org/10.1093/mnras/stz226} {\\bibfield  {journal} {\\bibinfo  {journal} "
        "{Mon. Not. Roy. Astron. Soc.}\\ }\\textbf {\\bibinfo {volume} {484}},\\ "
        "\\bibinfo {pages} {4216} (\\bibinfo {year} {2019})}\\BibitemShut {NoStop}%\n"
        "\\end{thebibliography}"
    )
    e = extract_bibitems_from_tex(tex)[0]
    assert e["title"] == "Constraints on binary black hole populations from LIGO\u2013Virgo detections"
    assert "\n" not in e["title"]
    assert e["year"] == 2019