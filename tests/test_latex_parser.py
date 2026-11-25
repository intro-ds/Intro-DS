from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.latexParser.runner import run_parser


def test_run_parser_two_pass_and_statuses(tmp_path: Path):
    source_dir = tmp_path / "sources"
    v1_dir = source_dir / "2607.23694" / "v1"
    v2_dir = source_dir / "2607.23694" / "v2"
    v3_empty_dir = source_dir / "2607.23695" / "v1"
    v1_dir.mkdir(parents=True)
    v2_dir.mkdir(parents=True)
    v3_empty_dir.mkdir(parents=True)

    (v1_dir / "main.tex").write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Intro}
        See \cite{old_key}.
        \end{document}
        """,
        encoding="utf-8",
    )
    (v2_dir / "main.tex").write_text(
        r"""
        \documentclass{article}
        \begin{document}
        \section{Intro}
        See \cite{new_key}.
        \end{document}
        """,
        encoding="utf-8",
    )

    source_manifest = tmp_path / "source_download_manifest.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arxiv_id", "version", "status"])
        writer.writeheader()
        writer.writerow({"arxiv_id": "2607.23694", "version": "1", "status": "success"})
        writer.writerow({"arxiv_id": "2607.23694", "version": "2", "status": "success"})
        writer.writerow({"arxiv_id": "2607.23695", "version": "1", "status": "no_source"})

    import src.latexParser.run as run_mod

    history_file = tmp_path / "arxiv_history_all_assignments.jsonl"
    history_file.write_text(
        json.dumps({"base_id": "2607.23694", "versions": ["2607.23694v1", "2607.23694v2"]}) + "\n"
        + json.dumps({"base_id": "2607.23695", "versions": ["2607.23695v1"]}) + "\n",
        encoding="utf-8",
    )
    orig_discovery = run_mod.DISCOVERY_FILE
    orig_raw = run_mod.RAW_DIR
    run_mod.DISCOVERY_FILE = history_file
    run_mod.RAW_DIR = tmp_path

    def fake_extract(compiled_tex: str, ver_dir: Path, arxiv_id: str = "", version: int = 1):
        if version == 1:
            return [{"citation_key": "old_key", "entry_type": "article", "title": "Paper A"}]
        return [{"citation_key": "new_key", "entry_type": "article", "title": "Paper A", "year": 2024}]

    def fake_dedup(raw_entries: list[dict], arxiv_id: str = ""):
        return (
            [{
                "arxiv_id": arxiv_id,
                "citation_key": "new_key",
                "entry_type": "article",
                "title": "Paper A",
                "authors": ["Alice"],
                "year": 2024,
                "doi": None,
                "journal": None,
                "url": None,
            }],
            {"old_key": "new_key", "new_key": "new_key"},
        )

    def fake_rewrite(compiled_tex: str, key_map: dict[str, str]) -> str:
        out = compiled_tex
        for old, new in key_map.items():
            out = out.replace(f"\\cite{{{old}}}", f"\\cite{{{new}}}")
        return out

    run_mod._resolve_bibtex_hooks = lambda: (fake_extract, fake_dedup, fake_rewrite)

    out_dir = tmp_path / "processed"
    try:
        processed = run_parser(
            base_ids=["2607.23694", "2607.23695"],
            source_dir=source_dir,
            source_manifest=source_manifest,
            output_dir=out_dir,
            follow=False,
        )
    finally:
        run_mod.DISCOVERY_FILE = orig_discovery
        run_mod.RAW_DIR = orig_raw

    assert processed == 2
    bib_table = pq.read_table(out_dir / "bibtex.parquet")
    assert pa.types.is_int32(bib_table.schema.field("year").type)
    assert pa.types.is_list(bib_table.schema.field("authors").type)

    with (out_dir / "parse_manifest.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["arxiv_id"], r["version"]): r for r in rows}
    assert by_key[("2607.23694", "1")]["status"] == "parsed"
    assert by_key[("2607.23694", "1")]["bibtex_raw_count"] == "1"
    assert by_key[("2607.23695", "1")]["status"] == "empty"