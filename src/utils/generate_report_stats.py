"""Crawl statistics for the Milestone 1 report (notebook 01 sections 3-5, poster).

Reads the manifests written by src.crawling.source_and_references and reports
coverage, success rates, source size before/after filtering and reference
counts. Every rate states its denominator.

Usage:
    python -m src.utils.generate_report_stats --student-id 24127590 \
        --output-json data/processed/crawl_stats.json \
        --problems-csv data/raw/problem_versions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from src.crawling.source_and_references import (
    SEMANTIC_SCHOLAR_MANIFEST,
    SOURCE_MANIFEST,
    _load_histories,
    _load_latest_status,
    assigned_base_ids,
)

SOURCE_OK = {"success"}
SOURCE_EMPTY = {"no_source", "no_tex_or_bib", "non_archive"}  # kept as empty records


def _pct(numerator: int, denominator: int) -> Optional[float]:
    return round(100 * numerator / denominator, 2) if denominator else None


def _avg(values: List[int]) -> Optional[float]:
    return round(mean(values), 2) if values else None


def compute_crawl_stats(
    base_ids: List[str],
    histories: Dict[str, List[str]],
    source_manifest: Path = SOURCE_MANIFEST,
    semantic_manifest: Path = SEMANTIC_SCHOLAR_MANIFEST,
) -> Dict[str, object]:
    """Compute paper-level, version-level, size and Semantic Scholar statistics."""
    source = _load_latest_status(Path(source_manifest), ("arxiv_id", "version"))
    semantic = _load_latest_status(Path(semantic_manifest), ("arxiv_id",))

    resolved = [b for b in base_ids if histories.get(b)]
    counts = {"success": 0, "empty": 0, "failed": 0}
    problems: List[dict] = []
    before: List[int] = []
    after: List[int] = []
    complete = with_tex = 0

    for base_id in resolved:
        statuses = []
        for version_id in histories[base_id]:
            version = version_id.rsplit("v", 1)[-1]
            row = source.get((base_id, version), {})
            status = row.get("status", "not_attempted")
            statuses.append(status)
            if status in SOURCE_OK:
                counts["success"] += 1
                before.append(int(row.get("source_bytes_before_filter") or 0))
                after.append(int(row.get("source_bytes") or 0))
            elif status in SOURCE_EMPTY:
                counts["empty"] += 1
            else:
                counts["failed"] += 1
                problems.append({
                    "arxiv_id": base_id, "version": version,
                    "status": status, "error": row.get("error", ""),
                })
        complete += all(s in SOURCE_OK | SOURCE_EMPTY for s in statuses)
        with_tex += any(s in SOURCE_OK for s in statuses)

    expected = sum(counts.values())
    rows = [semantic.get((b,), {}) for b in base_ids]
    ok_rows = [r for r in rows if r.get("status") == "success"]
    not_found = sum(r.get("status") == "not_found" for r in rows)
    before_total, after_total = sum(before), sum(after)

    return {
        "papers": {
            "assigned": len(base_ids), "resolved": len(resolved),
            "complete": complete, "with_tex": with_tex,
            "resolved_rate_pct": _pct(len(resolved), len(base_ids)),
            "complete_rate_pct": _pct(complete, len(resolved)),
            "with_tex_rate_pct": _pct(with_tex, len(resolved)),
        },
        "versions": {
            "expected": expected, **counts,
            "success_rate_pct": _pct(counts["success"], expected),
            "empty_rate_pct": _pct(counts["empty"], expected),
        },
        "size": {
            "avg_before_bytes": _avg(before), "avg_after_bytes": _avg(after),
            "reduction_pct": (
                round(100 * (1 - after_total / before_total), 2) if before_total else None
            ),
        },
        "semantic_scholar": {
            "success": len(ok_rows), "not_found": not_found,
            "failed": len(base_ids) - len(ok_rows) - not_found,
            "success_rate_pct": _pct(len(ok_rows), len(base_ids)),
            "avg_refs_per_paper": _avg([int(r.get("reference_count") or 0) for r in ok_rows]),
            "avg_arxiv_refs_per_paper": _avg(
                [int(r.get("arxiv_reference_count") or 0) for r in ok_rows]
            ),
        },
        "problem_versions": problems,
    }


def format_markdown(stats: Dict[str, object]) -> str:
    """Render the statistics as a Markdown table with explicit denominators."""
    p, v, s, r = (stats[k] for k in ("papers", "versions", "size", "semantic_scholar"))
    return "\n".join([
        "| Metric | Value | Denominator |",
        "|---|---:|---|",
        f"| Papers resolved | {p['resolved']} ({p['resolved_rate_pct']}%) | {p['assigned']} assigned papers |",
        f"| Papers fully crawled | {p['complete']} ({p['complete_rate_pct']}%) | {p['resolved']} resolved papers |",
        f"| Papers with TeX/BibTeX | {p['with_tex']} ({p['with_tex_rate_pct']}%) | {p['resolved']} resolved papers |",
        f"| Versions with TeX/BibTeX | {v['success']} ({v['success_rate_pct']}%) | {v['expected']} expected versions |",
        f"| Versions kept as empty records | {v['empty']} ({v['empty_rate_pct']}%) | {v['expected']} expected versions |",
        f"| Versions failed or pending | {v['failed']} | {v['expected']} expected versions |",
        f"| Avg source size before / after filter (bytes) | {s['avg_before_bytes']} / {s['avg_after_bytes']} ({s['reduction_pct']}% smaller) | successful versions |",
        f"| Semantic Scholar success | {r['success']} ({r['success_rate_pct']}%) | {p['assigned']} assigned papers |",
        f"| Semantic Scholar not found / failed | {r['not_found']} / {r['failed']} | {p['assigned']} assigned papers |",
        f"| Avg references (all / with arXiv ID) per paper | {r['avg_refs_per_paper']} / {r['avg_arxiv_refs_per_paper']} | papers with S2 success |",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Milestone 1 crawl statistics.")
    ap.add_argument("--student-id", help="restrict to one member's range; default: all members")
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--problems-csv", type=Path, help="write failed/pending versions here")
    args = ap.parse_args(argv)

    base_ids = assigned_base_ids(student_id=args.student_id)
    stats = compute_crawl_stats(base_ids, _load_histories(base_ids))
    problems = stats.pop("problem_versions")
    print(format_markdown(stats))
    print(f"\nFailed or pending versions: {len(problems)}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if args.problems_csv:
        args.problems_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.problems_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["arxiv_id", "version", "status", "error"])
            writer.writeheader()
            writer.writerows(problems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())