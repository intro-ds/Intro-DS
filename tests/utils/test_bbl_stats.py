"""Unit tests for the bbl_stats utility."""

import json
from pathlib import Path

from src.utils.bbl_stats import compute_bbl_stats, is_bbl_only


def test_is_bbl_only(tmp_path: Path) -> None:
    """Test detection of versions that only have a compiled .bbl."""
    (tmp_path / "refs.bbl").write_text("\\bibitem{a} A.", encoding="utf-8")
    (tmp_path / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    assert is_bbl_only(tmp_path) is True

    (tmp_path / "refs.bib").write_text("@article{a,}", encoding="utf-8")
    assert is_bbl_only(tmp_path) is False


def test_compute_bbl_stats(tmp_path: Path) -> None:
    """Test counting .bbl-only papers and their parse rate."""
    corpus, inter = tmp_path / "corpus", tmp_path / "intermediate"
    v1 = corpus / "2512.00001" / "v1"
    v1.mkdir(parents=True)
    (v1 / "refs.bbl").write_text("\\bibitem{a} A.", encoding="utf-8")
    (inter / "2512.00001").mkdir(parents=True)
    (inter / "2512.00001" / "parsed.json").write_text(
        json.dumps({"bibtex": [{"citation_key": "a"}]}), encoding="utf-8"
    )

    res = compute_bbl_stats(corpus, inter)
    assert res == {"papers": 1, "bbl_only": 1, "bbl_only_with_bibtex": 1}