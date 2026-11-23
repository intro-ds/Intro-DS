"""Sanity checks for the Milestone 2 Parquet tables and per-version parse status.

Usage:
    python -m src.utils.stats [--student-id 24127590]
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from src.crawling.config_reader import load_member_range


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Milestone 2 table sanity checks")
    parser.add_argument("--student-id", default="24127590")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    args = parser.parse_args()

    d = args.processed_dir / args.student_id
    el, hi, bi = (pd.read_parquet(d / f"{t}.parquet") for t in ("elements", "hierarchy", "bibtex"))
    assigned = load_member_range(args.student_id)["arxiv_ids"]

    print(f"assigned papers: {len(assigned)}")
    print(f"papers in elements: {el.arxiv_id.nunique()} | hierarchy: {hi.arxiv_id.nunique()} "
          f"| bibtex: {bi.arxiv_id.nunique()}")
    print("element_id duplicated (expect 0):", el.element_id.duplicated().sum())
    print("hierarchy element_id not in elements (expect 0):", (~hi.element_id.isin(el.element_id)).sum())
    print("hierarchy parent_id not in elements (expect 0):", (~hi.parent_id.isin(el.element_id)).sum())
    print("self-parent edges (expect 0):", (hi.element_id == hi.parent_id).sum())

    multi = hi.groupby(["arxiv_id", "version", "element_id"]).parent_id.nunique()
    bad = multi[multi > 1].reset_index()
    print(f"nodes with >1 parent in a version (expect 0): {len(bad)} "
          f"in {bad.arxiv_id.nunique()} papers")

    # Status of every version, and why some papers have no elements
    version_status: Counter = Counter()
    paper_reasons: Counter = Counter()
    without_elements = sorted(set(assigned) - set(el.arxiv_id))
    missing_json = []
    for aid in assigned:
        f = args.intermediate_dir / aid / "parsed.json"
        if not f.exists():
            missing_json.append(aid)
            continue
        versions = json.loads(f.read_text(encoding="utf-8")).get("versions", {})
        reasons = []
        for meta in versions.values():
            key = "parsed" if meta.get("parsed") else str(meta.get("reason", "unknown")).split(":")[0]
            version_status[key] += 1
            reasons.append(key)
        if aid in set(without_elements):
            paper_reasons[",".join(sorted(set(reasons))) or "no_versions"] += 1

    print("version status:", dict(version_status))
    print(f"papers without elements: {len(without_elements)} | by reason: {dict(paper_reasons)}")
    print(f"papers without parsed.json: {len(missing_json)}")

    out = d / "papers_without_elements.txt"
    out.write_text("\n".join(without_elements), encoding="utf-8")
    print(f"list written to {out}")


if __name__ == "__main__":
    main()