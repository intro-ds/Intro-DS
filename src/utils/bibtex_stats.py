"""Report how many unified BibTeX entries miss key fields.

Usage:
    python -m src.utils.bibtex_stats [--intermediate-dir data/intermediate]
"""

import argparse
import json
from pathlib import Path

FIELDS = ("title", "authors", "year", "doi", "journal", "url")


def compute_missing_rates(intermediate_dir: Path) -> dict:
    """Return the number of entries and the missing rate (%) of each field."""
    total = 0
    missing = {f: 0 for f in FIELDS}
    papers_without_bib = 0
    files = sorted(intermediate_dir.glob("*/parsed.json"))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        entries = data.get("bibtex", [])
        if not entries:
            papers_without_bib += 1
        for e in entries:
            total += 1
            for field in FIELDS:
                if not e.get(field):
                    missing[field] += 1
    rates = {f: round(100 * missing[f] / total, 2) if total else 0.0 for f in FIELDS}
    return {
        "papers": len(files),
        "papers_without_bibtex": papers_without_bib,
        "entries": total,
        "missing_rate_pct": rates,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="BibTeX field coverage report")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    args = parser.parse_args()
    res = compute_missing_rates(args.intermediate_dir)
    print(f"papers={res['papers']} without_bibtex={res['papers_without_bibtex']} entries={res['entries']}")
    for field, rate in res["missing_rate_pct"].items():
        print(f"  missing {field:8s} {rate:6.2f}%")


if __name__ == "__main__":
    main()