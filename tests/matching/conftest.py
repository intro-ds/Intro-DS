"""Shared fixtures for the matching tests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


class Workspace:
    """Temporary intermediate/corpus/output directories with fake publications."""

    def __init__(self, root: Path) -> None:
        self.intermediate_dir = root / "intermediate"
        self.corpus_dir = root / "corpus"
        self.output_dir = root / "matching"

    @staticmethod
    def entry(
        key: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build a unified BibTeX entry dict."""
        data = {
            "citation_key": key,
            "entry_type": "article",
            "title": title,
            "authors": authors or [],
            "year": year,
            "doi": None,
            "journal": None,
            "url": None,
        }
        data.update(extra)
        return data

    @staticmethod
    def ref(
        arxiv_id: Optional[str],
        title: str,
        authors: Optional[List[str]] = None,
        date: str = "",
    ) -> Dict[str, Any]:
        """Build a crawled reference dict."""
        return {
            "ref_s2_id": f"s2_{arxiv_id}",
            "ref_arxiv_id": arxiv_id,
            "ref_title": title,
            "ref_authors": authors or [],
            "ref_submission_date": date,
        }

    def add_publication(
        self, arxiv_id: str, entries: List[Dict[str, Any]], refs: List[Dict[str, Any]]
    ) -> None:
        """Write parsed.json and refs.json for one fake publication."""
        pdir = self.intermediate_dir / arxiv_id
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "parsed.json").write_text(
            json.dumps({"arxiv_id": arxiv_id, "bibtex": entries}), encoding="utf-8"
        )
        cdir = self.corpus_dir / arxiv_id
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "refs.json").write_text(
            json.dumps({"arxiv_id": arxiv_id, "references": refs}), encoding="utf-8"
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Provide an empty Workspace."""
    return Workspace(tmp_path)


PAPERS = [
    ("1706.03762", "Attention Is All You Need", ["Ashish Vaswani", "Noam Shazeer"], "2017-06-12"),
    ("1512.03385", "Deep Residual Learning for Image Recognition", ["Kaiming He", "Xiangyu Zhang"], "2015-12-10"),
    ("1412.6980", "Adam: A Method for Stochastic Optimization", ["Diederik Kingma", "Jimmy Ba"], "2014-12-22"),
    ("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers", ["Jacob Devlin", "Ming-Wei Chang"], "2018-10-11"),
    ("1409.1556", "Very Deep Convolutional Networks for Large-Scale Image Recognition", ["Karen Simonyan", "Andrew Zisserman"], "2014-09-04"),
]


@pytest.fixture
def matched_publication(workspace: Workspace):
    """Factory: add a publication whose i-th entry corresponds to the i-th reference."""

    def _add(arxiv_id: str) -> None:
        entries = [
            Workspace.entry(f"key{i}", title=t, authors=[a.split()[-1] for a in au], year=int(d[:4]))
            for i, (_, t, au, d) in enumerate(PAPERS)
        ]
        refs = [Workspace.ref(rid, t, au, d) for rid, t, au, d in PAPERS]
        workspace.add_publication(arxiv_id, entries, refs)

    return _add