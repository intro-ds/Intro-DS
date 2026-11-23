"""Check completeness of the crawled corpus for one student's ID range.

Reports, per assigned paper: metadata present, every version has TeX/BibTeX
sources (or an explicit .no_latex marker), and refs.json exists.
Missing items are written to data/raw/missing_*.txt so the crawler can be re-run.

Usage:
    python -m src.utils.validate_corpus --student-id 24127590
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List

from src.crawling.config_reader import load_member_range


def validate_corpus(
    student_id: str,
    corpus_dir: Path = Path("data/raw/corpus"),
    report_dir: Path = Path("data/raw"),
) -> Dict[str, object]:
    """Validate corpus coverage and write lists of missing ids."""
    ids: List[str] = load_member_range(student_id)["arxiv_ids"]
    stat: Counter = Counter()
    missing: Dict[str, List[str]] = {"metadata": [], "versions": [], "refs": []}

    for aid in ids:
        pdir = corpus_dir / aid
        meta_file = pdir / "metadata.json"
        if not meta_file.exists():
            missing["metadata"].append(aid)
            continue
        stat["papers_with_metadata"] += 1

        versions = json.loads(meta_file.read_text(encoding="utf-8")).get("versions", [])
        for idx in range(1, len(versions) + 1):
            stat["versions_expected"] += 1
            vdir = pdir / f"v{idx}"
            if any(vdir.glob("*.tex")) or any(vdir.glob("*.bib")):
                stat["versions_with_tex"] += 1
            elif (vdir / ".no_latex").exists():
                stat["versions_no_latex"] += 1
            else:
                missing["versions"].append(f"{aid}v{idx}")

        refs_file = pdir / "refs.json"
        if not refs_file.exists():
            missing["refs"].append(aid)
            continue
        stat["papers_with_refs"] += 1
        refs = json.loads(refs_file.read_text(encoding="utf-8"))
        n_arxiv = sum(1 for r in refs.get("references", []) if r.get("ref_arxiv_id"))
        stat["refs_total"] += len(refs.get("references", []))
        stat["refs_with_arxiv_id"] += n_arxiv
        if refs.get("note") == "not_found_on_s2":
            stat["papers_not_on_s2"] += 1

    report_dir.mkdir(parents=True, exist_ok=True)
    for name, items in missing.items():
        (report_dir / f"missing_{name}.txt").write_text("\n".join(items), encoding="utf-8")

    return {"assigned": len(ids), **stat, **{f"missing_{k}": len(v) for k, v in missing.items()}}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate crawled corpus coverage")
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw/corpus"))
    args = parser.parse_args()
    for key, val in validate_corpus(args.student_id, args.corpus_dir).items():
        print(f"{key:28s} {val}")


if __name__ == "__main__":
    main()