import pytest

from src.latexParser import PublicationHierarchyBuilder

split = PublicationHierarchyBuilder.split_into_sentences


def test_split_keeps_abbreviations_together():
    cases = {
        "We use a simple model, i.e. a linear one. It works well.":
            ["We use a simple model, i.e. a linear one.", "It works well."],
        "Many tasks, e.g. parsing, are hard. We study them.":
            ["Many tasks, e.g. parsing, are hard.", "We study them."],
        "Smith et al. proposed a method. It is fast.":
            ["Smith et al. proposed a method.", "It is fast."],
        "This follows from Eq. (3) and Fig. 2. The proof is short.":
            ["This follows from Eq. (3) and Fig. 2.", "The proof is short."],
    }
    for text, expected in cases.items():
        assert split(text) == expected, text


def test_split_protects_decimals_urls_and_math():
    assert split("We get 0.05 and 3.14. Then we stop.") == [
        "We get 0.05 and 3.14.", "Then we stop."]
    assert split("See https://example.com/a.b for details. It is open.") == [
        "See https://example.com/a.b for details.", "It is open."]
    # a dot followed by a capital letter inside $...$ must not create a new leaf
    assert split("Let $a. B$ be a set. Then $x = 1.5$ holds.") == [
        "Let $a. B$ be a set.", "Then $x = 1.5$ holds."]


@pytest.mark.xfail(
    reason="BUG structure_extractor.split_into_sentences: URL regex swallows the "
           "trailing period, so a sentence ending with a URL is not split",
    strict=False,
)
def test_split_url_at_end_of_sentence():
    assert split("Code is at https://github.com/x/y. It is open.") == [
        "Code is at https://github.com/x/y.", "It is open."]