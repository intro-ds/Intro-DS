"""Unit tests for the clean_control_entries utility."""

import json
from pathlib import Path

from src.utils.clean_control_entries import clean_intermediate


def _write(tmp_path: Path) -> Path:
    pdir = tmp_path / "2511.19127"
    pdir.mkdir()
    path = pdir / "parsed.json"
    path.write_text(
        json.dumps({
            "arxiv_id": "2511.19127",
            "bibtex": [
                {"citation_key": "ctrl", "entry_type": "control"},
                {"citation_key": "real", "entry_type": "article"},
            ],
        }),
        encoding="utf-8",
    )
    return path


def test_dry_run_does_not_modify(tmp_path: Path) -> None:
    path = _write(tmp_path)
    res = clean_intermediate(tmp_path, dry_run=True)
    assert res == {"papers_scanned": 1, "papers_changed": 1, "entries_removed": 1}
    assert len(json.loads(path.read_text(encoding="utf-8"))["bibtex"]) == 2


def test_clean_removes_only_control_entries(tmp_path: Path) -> None:
    path = _write(tmp_path)
    clean_intermediate(tmp_path)
    keys = [e["citation_key"] for e in json.loads(path.read_text(encoding="utf-8"))["bibtex"]]
    assert keys == ["real"]