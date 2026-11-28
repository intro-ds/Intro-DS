"""Unit tests for the crawl statistics report."""
import csv
from pathlib import Path

from src.utils.generate_report_stats import compute_crawl_stats, format_markdown


def _write_csv(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compute_crawl_stats(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    semantic = tmp_path / "semantic.csv"
    _write_csv(source, [
        {"arxiv_id": "2511.00001", "version": "1", "status": "success",
         "source_bytes_before_filter": "1000", "source_bytes": "200", "error": ""},
        {"arxiv_id": "2511.00001", "version": "2", "status": "non_archive",
         "source_bytes_before_filter": "0", "source_bytes": "0", "error": ""},
        {"arxiv_id": "2511.00002", "version": "1", "status": "error",
         "source_bytes_before_filter": "0", "source_bytes": "0", "error": "HTTP 503"},
    ])
    _write_csv(semantic, [
        {"arxiv_id": "2511.00001", "status": "success",
         "reference_count": "10", "arxiv_reference_count": "4"},
        {"arxiv_id": "2511.00002", "status": "not_found",
         "reference_count": "0", "arxiv_reference_count": "0"},
    ])
    histories = {
        "2511.00001": ["2511.00001v1", "2511.00001v2"],
        "2511.00002": ["2511.00002v1"],
    }

    stats = compute_crawl_stats(
        ["2511.00001", "2511.00002", "2511.00003"], histories, source, semantic
    )

    assert stats["papers"]["assigned"] == 3
    assert stats["papers"]["resolved"] == 2
    assert stats["papers"]["complete"] == 1
    assert stats["papers"]["with_tex"] == 1
    assert stats["versions"]["expected"] == 3
    assert (stats["versions"]["success"], stats["versions"]["empty"],
            stats["versions"]["failed"]) == (1, 1, 1)
    assert stats["size"]["reduction_pct"] == 80.0
    assert stats["semantic_scholar"]["success"] == 1
    assert stats["semantic_scholar"]["not_found"] == 1
    assert stats["semantic_scholar"]["failed"] == 1
    assert stats["semantic_scholar"]["avg_refs_per_paper"] == 10.0
    assert stats["problem_versions"][0]["error"] == "HTTP 503"
    assert "| Metric | Value | Denominator |" in format_markdown(
        {k: v for k, v in stats.items() if k != "problem_versions"}
    )