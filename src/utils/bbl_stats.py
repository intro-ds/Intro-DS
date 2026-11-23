"""Count papers whose references exist only as a compiled .bbl, and how many parse.

A paper is ".bbl-only" when its latest version has a .bbl file but no .bib file
and no \\bibitem inside its .tex files.

Usage:
    python -m src.utils.bbl_stats [--corpus-dir data/raw/corpus]
                                  [--intermediate-dir data/intermediate]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


def is_bbl_only(version_dir: Path) -> bool:
    """Return True if the version keeps references only as a .bbl file."""
    if not any(version_dir.glob("*.bbl")) or any(version_dir.glob("*.bib")):
        return False
    for tex in version_dir.glob("*.tex"):
        if "\\bibitem" in tex.read_text(encoding="utf-8", errors="replace"):
            return False
    return True


def _latest_version_dir(paper_dir: Path) -> Optional[Path]:
    """Return the vN directory with the highest N, or None."""
    versions = [
        (int(m.group(1)), d)
        for d in paper_dir.glob("v*")
        if d.is_dir() and (m := re.match(r"^v(\d+)$", d.name))
    ]
    return max(versions, key=lambda x: x[0])[1] if versions else None


def compute_bbl_stats(corpus_dir: Path, intermediate_dir: Path) -> Dict[str, int]:
    """Count .bbl-only papers and those that yielded at least one BibTeX entry."""
    papers = bbl_only = parsed_ok = 0
    for paper_dir in sorted(corpus_dir.glob("*")):
        if not paper_dir.is_dir() or paper_dir.name.startswith("."):
            continue
        latest = _latest_version_dir(paper_dir)
        if latest is None:
            continue
        papers += 1
        if not is_bbl_only(latest):
            continue
        bbl_only += 1
        parsed = intermediate_dir / paper_dir.name / "parsed.json"
        if parsed.exists():
            data = json.loads(parsed.read_text(encoding="utf-8"))
            if data.get("bibtex"):
                parsed_ok += 1
    return {"papers": papers, "bbl_only": bbl_only, "bbl_only_with_bibtex": parsed_ok}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=".bbl-only paper statistics")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw/corpus"))
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    args = parser.parse_args()
    res = compute_bbl_stats(args.corpus_dir, args.intermediate_dir)
    rate = 100 * res["bbl_only_with_bibtex"] / res["bbl_only"] if res["bbl_only"] else 0.0
    print(f"papers={res['papers']} bbl_only={res['bbl_only']} parsed_with_bibtex={res['bbl_only_with_bibtex']}")
    print(f".bbl-only parse rate: {rate:.2f}%")


if __name__ == "__main__":
    main()