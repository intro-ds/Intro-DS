import pytest

from src.latexParser.bibtex_dedup import dedup_bibtex, rewrite_cites


def test_deduplicates_example_across_versions_and_unions_fields():
    unified, key_map = dedup_bibtex(
        {
            1: [
                {
                    "citation_key": "vaswani2017",
                    "title": "Attention Is All You Need",
                    "authors": "Ashish Vaswani, et al.",
                    "year": "2017",
                }
            ],
            2: [
                {
                    "citation_key": "vit_transformer",
                    "title": "Attention is All You Need",
                    "authors": "Ashish Vaswani, et al.",
                    "year": "2017",
                    "doi": "https://doi.org/10.48550/ARXIV.1706.03762",
                }
            ],
            3: [
                {
                    "citation_key": "vit_transformer",
                    "title": "Attention Is All You Need",
                    "authors": "Ashish Vaswani, et al.",
                    "doi": "10.48550/arXiv.1706.03762",
                }
            ],
        }
    )

    assert len(unified) == 1
    assert unified[0] == {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, et al.",
        "year": "2017",
        "doi": "https://doi.org/10.48550/ARXIV.1706.03762",
        "citation_key": "vaswani2017",
    }
    assert key_map == {
        "vaswani2017": "vaswani2017",
        "vit_transformer": "vaswani2017",
    }
    rewritten = rewrite_cites(
        r"\cite{vit_transformer, vaswani2017}", key_map
    )
    valid_keys = {entry["citation_key"] for entry in unified}
    cited_keys = {
        key.strip()
        for key in rewritten.removeprefix(r"\cite{").removesuffix("}").split(",")
    }
    assert cited_keys <= valid_keys


def test_matches_normalized_title_and_year_without_doi():
    unified, key_map = dedup_bibtex(
        {
            1: [{"citation_key": "old", "title": "{A} Study: On Things!", "year": 2020}],
            2: [{"citation_key": "new", "title": "a study on things", "year": "2020"}],
        }
    )

    assert len(unified) == 1
    assert key_map["new"] == "old"


def test_different_nonempty_dois_keep_same_title_and_year_separate():
    unified, key_map = dedup_bibtex(
        {
            1: [
                {"citation_key": "paper_a", "title": "Same title", "year": "2020", "doi": "10.1/a"},
                {"citation_key": "paper_b", "title": "Same title", "year": "2020", "doi": "10.1/b"},
            ]
        }
    )

    assert len(unified) == 2
    assert key_map == {"paper_a": "paper_a", "paper_b": "paper_b"}


def test_conflicting_field_prefers_more_complete_value():
    unified, _ = dedup_bibtex(
        {
            1: [{"citation_key": "old", "title": "Title", "year": "2020", "authors": "A. Author"}],
            2: [{"citation_key": "new", "title": "Title", "year": "2020", "authors": "Alexandra Author"}],
        }
    )

    assert unified[0]["authors"] == "Alexandra Author"


def test_rewrites_all_cite_commands_and_comma_separated_keys():
    text = (
        r"\cite{vaswani2017, vit_transformer}"
        r" \citep[p.1]{vit_transformer}"
        r" \citet[see][p. 2]{vaswani2017}"
        r" \citeauthor*{vit_transformer}"
    )

    rewritten = rewrite_cites(
        text,
        {"vit_transformer": "vaswani2017", "vaswani2017": "canonical"},
    )

    assert rewritten == (
        r"\cite{canonical, canonical}"
        r" \citep[p.1]{canonical}"
        r" \citet[see][p. 2]{canonical}"
        r" \citeauthor*{canonical}"
    )


def test_rejects_cyclic_key_mapping():
    with pytest.raises(ValueError, match="Cyclic citation-key mapping"):
        rewrite_cites(r"\cite{a}", {"a": "b", "b": "a"})
