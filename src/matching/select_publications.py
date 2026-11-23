"""Publication selection for manual labelling (Milestone 3) and shared data loaders.

Reads the parsed BibTeX entries (data/intermediate/<arxiv_id>/parsed.json) and the
crawled Semantic Scholar references (data/raw/corpus/<arxiv_id>/refs.json), computes
per-publication statistics, and picks the publications that are the best candidates
for manual labelling: many BibTeX entries with a title AND many references that carry
an arXiv ID (so that positive matches actually exist).

Usage:
    python -m src.matching.select_publications --student-id 24127590 --num 7

Outputs (default directory data/interim/matching/):
    publication_stats.csv        statistics of every parsed publication
    manual_publications.txt      selected arXiv IDs, one per line
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from src.crawling.config_reader import load_member_range

DEFAULT_STUDENT_ID = "24127590"
DEFAULT_INTERMEDIATE_DIR = Path("data/intermediate")
DEFAULT_CORPUS_DIR = Path("data/raw/corpus")
DEFAULT_OUTPUT_DIR = Path("data/interim/matching")
MANUAL_PUBLICATIONS_FILE = "manual_publications.txt"
STATS_FILE = "publication_stats.csv"
STATS_COLUMNS = [
    "arxiv_id",
    "n_bibtex",
    "n_bibtex_with_title",
    "n_arxiv_refs",
    "expected_matchable",
]


# ---------------------------------------------------------------------------
# Shared data loaders (also used by build_candidates.py and auto_labeler.py)
# ---------------------------------------------------------------------------


def load_entries(arxiv_id: str, intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR) -> List[Dict[str, Any]]:
    """Load the unified BibTeX entries of one publication from parsed.json.

    Returns an empty list when the file is missing or unreadable.
    """
    path = intermediate_dir / arxiv_id / "parsed.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return list(data.get("bibtex") or [])


def load_references(arxiv_id: str, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> List[Dict[str, Any]]:
    """Load the crawled references of one publication that carry an arXiv ID.

    References without an arXiv ID are dropped (they are not rows of the published
    `references` table) and duplicated arXiv IDs are kept once.
    """
    path = corpus_dir / arxiv_id / "refs.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    seen: Set[str] = set()
    refs: List[Dict[str, Any]] = []
    for raw in data.get("references") or []:
        ref_id = raw.get("ref_arxiv_id")
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        refs.append(
            {
                "ref_arxiv_id": str(ref_id),
                "ref_title": raw.get("ref_title") or "",
                "ref_authors": list(raw.get("ref_authors") or []),
                "ref_submission_date": raw.get("ref_submission_date") or "",
                "ref_s2_id": raw.get("ref_s2_id") or "",
            }
        )
    return refs


def list_publications(
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    student_id: Optional[str] = None,
) -> List[str]:
    """List arXiv IDs that have a parsed.json, optionally restricted to one member's range."""
    if student_id:
        candidates = load_member_range(student_id)["arxiv_ids"]
        return [a for a in candidates if (intermediate_dir / a / "parsed.json").exists()]
    return sorted(
        p.name for p in intermediate_dir.glob("*") if (p / "parsed.json").exists()
    )


def read_id_file(path: Path) -> List[str]:
    """Read arXiv IDs from a text file (one per line, '#' lines and blanks ignored)."""
    if not path.exists():
        return []
    ids: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


# ---------------------------------------------------------------------------
# Statistics and selection
# ---------------------------------------------------------------------------


def compute_publication_stats(
    arxiv_ids: List[str],
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """Compute BibTeX / reference counts for each publication.

    expected_matchable = min(entries with a title, references with an arXiv ID): an upper
    bound of the number of positive (entry, reference) pairs a publication can offer.
    """
    stats: List[Dict[str, Any]] = []
    iterator = tqdm(arxiv_ids, desc="Computing publication stats", unit="paper") if show_progress else arxiv_ids
    for aid in iterator:
        entries = load_entries(aid, intermediate_dir)
        refs = load_references(aid, corpus_dir)
        n_titled = sum(1 for e in entries if (e.get("title") or "").strip())
        stats.append(
            {
                "arxiv_id": aid,
                "n_bibtex": len(entries),
                "n_bibtex_with_title": n_titled,
                "n_arxiv_refs": len(refs),
                "expected_matchable": min(n_titled, len(refs)),
            }
        )
    return stats


def select_publications(
    stats: List[Dict[str, Any]],
    num: int = 6,
    min_bibtex: int = 20,
    min_arxiv_refs: int = 10,
    max_bibtex: int = 150,
) -> List[Dict[str, Any]]:
    """Pick the best publications for manual labelling.

    A publication is eligible when it has at least `min_bibtex` entries WITH a title,
    at most `max_bibtex` entries (to keep labelling feasible) and at least
    `min_arxiv_refs` references with an arXiv ID. Eligible publications are ranked by
    expected_matchable (descending), ties broken by arXiv ID.
    """
    eligible = [
        s
        for s in stats
        if s["n_bibtex_with_title"] >= min_bibtex
        and s["n_arxiv_refs"] >= min_arxiv_refs
        and s["n_bibtex"] <= max_bibtex
    ]
    eligible.sort(key=lambda s: (-s["expected_matchable"], s["arxiv_id"]))
    return eligible[:num]


def write_stats_csv(stats: List[Dict[str, Any]], path: Path) -> None:
    """Write publication statistics to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_COLUMNS)
        writer.writeheader()
        writer.writerows(stats)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Select publications for manual labelling")
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num", type=int, default=7, help="Publications to select (>=5 required, keep spares)")
    parser.add_argument("--min-bibtex", type=int, default=20, help="Min entries with a title")
    parser.add_argument("--min-arxiv-refs", type=int, default=10)
    parser.add_argument("--max-bibtex", type=int, default=150)
    args = parser.parse_args()

    ids = list_publications(args.intermediate_dir, args.student_id)
    print(f"Parsed publications found: {len(ids)}")
    stats = compute_publication_stats(ids, args.intermediate_dir, args.corpus_dir, show_progress=True)
    write_stats_csv(stats, args.output_dir / STATS_FILE)

    selected = select_publications(
        stats, args.num, args.min_bibtex, args.min_arxiv_refs, args.max_bibtex
    )
    manual_path = args.output_dir / MANUAL_PUBLICATIONS_FILE
    manual_path.parent.mkdir(parents=True, exist_ok=True)
    manual_path.write_text("\n".join(s["arxiv_id"] for s in selected) + "\n", encoding="utf-8")

    print(f"\nSelected {len(selected)} publication(s) -> {manual_path}")
    for s in selected:
        print(
            f"  {s['arxiv_id']}  bibtex={s['n_bibtex']:4d}  titled={s['n_bibtex_with_title']:4d}"
            f"  arxiv_refs={s['n_arxiv_refs']:4d}  expected_matchable={s['expected_matchable']:4d}"
        )
    if len(selected) < 5:
        print("WARNING: fewer than 5 eligible publications; lower --min-bibtex / --min-arxiv-refs.")


if __name__ == "__main__":
    main()