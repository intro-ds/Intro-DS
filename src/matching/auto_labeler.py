"""Rule-based automatic matcher (regex + string similarity) for Milestone 3.

For every BibTeX entry of a publication, finds the crawled reference (arXiv-keyed) of the
SAME publication it corresponds to, or none. Matching rules, in order:
  1. arXiv-ID rule: the entry's url / doi / journal / title mentions the arXiv ID of one
     of the references (e.g. "arXiv:1706.03762", "arxiv.org/abs/1706.03762v2").
  2. Similarity rule: weighted combination of title similarity (0.7), author last-name
     overlap (0.2) and publication-year proximity (0.1); the best candidate is accepted
     when its score >= threshold and it beats the runner-up by >= margin.

The same scoring is reused by build_candidates.py to suggest candidates for manual labelling.

Usage:
    python -m src.matching.auto_labeler --student-id 24127590 \
        --manual-file data/interim/matching/manual_publications.txt --fraction 0.10

Outputs (default directory data/interim/matching/):
    labels_auto.csv         arxiv_id, citation_key, groundtruth_arxiv_id, score, method, source
    auto_publications.txt   publications labelled automatically
"""

import argparse
import csv
import math
import random
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from src.matching.select_publications import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_INTERMEDIATE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STUDENT_ID,
    MANUAL_PUBLICATIONS_FILE,
    compute_publication_stats,
    list_publications,
    load_entries,
    load_references,
    read_id_file,
)

AUTO_LABELS_FILE = "labels_auto.csv"
AUTO_PUBLICATIONS_FILE = "auto_publications.txt"
LABEL_COLUMNS = ["arxiv_id", "citation_key", "groundtruth_arxiv_id", "score", "method", "source"]

WEIGHTS = {"title": 0.7, "author": 0.2, "year": 0.1}
UNKNOWN_TITLE_PENALTY = 0.85
DEFAULT_THRESHOLD = 0.80
DEFAULT_MARGIN = 0.05

STOP_WORDS = frozenset(
    "a an the of for in on and or to with by from at as is are via using use its their".split()
)
_AUTHOR_NOISE = frozenset({"al", "others", "et"})


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=200_000)
def normalize_text(text: Optional[str]) -> str:
    """Lowercase, strip accents and leftover LaTeX commands, keep only [a-z0-9 ]."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text.lower())
    return text.strip()


@lru_cache(maxsize=200_000)
def _title_tokens(text: Optional[str]) -> FrozenSet[str]:
    """Token set of a title without stop words."""
    return frozenset(t for t in normalize_text(text).split() if t not in STOP_WORDS)


@lru_cache(maxsize=100_000)
def _last_names(authors: Tuple[str, ...]) -> Tuple[str, ...]:
    """Ordered normalised last names; handles 'Last, F.' and 'First Last' forms."""
    names: List[str] = []
    for author in authors:
        raw = (author or "").strip()
        if not raw:
            continue
        last = raw.split(",")[0] if "," in raw else (raw.split() or [""])[-1]
        name = normalize_text(last).replace(" ", "")
        if len(name) >= 2 and name not in _AUTHOR_NOISE:
            names.append(name)
    return tuple(names)


# ---------------------------------------------------------------------------
# Component similarities (None = unknown, the component is ignored)
# ---------------------------------------------------------------------------


def title_similarity(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """Title similarity in [0, 1]; None when either title is empty."""
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return None
    ta, tb = _title_tokens(a), _title_tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    score = jaccard
    if jaccard >= 0.15:  # character-level ratio only for plausible pairs (speed)
        score = max(score, SequenceMatcher(None, na, nb).ratio())
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(small) >= 3 and small <= large:  # truncated title / title embedded in raw text
        score = max(score, 0.9)
    return score


def author_similarity(a: Optional[List[str]], b: Optional[List[str]]) -> Optional[float]:
    """Last-name overlap coefficient, at least 0.7 when the first authors agree."""
    la, lb = _last_names(tuple(a or ())), _last_names(tuple(b or ()))
    if not la or not lb:
        return None
    overlap = len(set(la) & set(lb)) / min(len(set(la)), len(set(lb)))
    if la[0] == lb[0]:
        overlap = max(overlap, 0.7)
    return overlap


def year_similarity(entry_year: Any, ref_date: Optional[str]) -> Optional[float]:
    """Year proximity: 1.0 same year, 0.7 off by one, 0.3 off by two, else 0."""
    match = re.match(r"(\d{4})", str(ref_date or ""))
    try:
        year = int(entry_year)
    except (TypeError, ValueError):
        return None
    if not match:
        return None
    return {0: 1.0, 1: 0.7, 2: 0.3}.get(abs(year - int(match.group(1))), 0.0)


def find_arxiv_id_match(entry: Dict[str, Any], ref_ids: List[str]) -> Optional[str]:
    """Return the reference arXiv ID explicitly mentioned in the entry's text fields."""
    blob = " ".join(str(entry.get(k) or "") for k in ("url", "doi", "journal", "title")).lower()
    if "arxiv" not in blob:
        return None
    for ref_id in ref_ids:
        pattern = r"(?<!\d)" + re.escape(ref_id.lower()) + r"(?:v\d+)?(?!\d)"
        if re.search(pattern, blob):
            return ref_id
    return None


def combine_scores(title: Optional[float], author: Optional[float], year: Optional[float]) -> float:
    """Weighted mean over the known components (weights renormalised).

    Needs a title or an author signal (year alone is meaningless). A missing title
    multiplies the score by UNKNOWN_TITLE_PENALTY.
    """
    if title is None and author is None:
        return 0.0
    parts = {"title": title, "author": author, "year": year}
    known = {k: v for k, v in parts.items() if v is not None}
    score = sum(WEIGHTS[k] * v for k, v in known.items()) / sum(WEIGHTS[k] for k in known)
    return score * UNKNOWN_TITLE_PENALTY if title is None else score


def score_pair(entry: Dict[str, Any], ref: Dict[str, Any], id_match: bool = False) -> Dict[str, Any]:
    """Score one (BibTeX entry, reference) pair."""
    title = title_similarity(entry.get("title"), ref.get("ref_title"))
    author = author_similarity(entry.get("authors"), ref.get("ref_authors"))
    year = year_similarity(entry.get("year"), ref.get("ref_submission_date"))
    total = 1.0 if id_match else combine_scores(title, author, year)
    return {
        "ref_arxiv_id": ref["ref_arxiv_id"],
        "ref_title": ref.get("ref_title", ""),
        "ref_authors": ref.get("ref_authors", []),
        "score": round(total, 4),
        "title_sim": title,
        "author_sim": author,
        "year_sim": year,
        "id_match": id_match,
    }


def rank_candidates(
    entry: Dict[str, Any], refs: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Rank the references of the publication for one entry (best first, top_k kept)."""
    matched_id = find_arxiv_id_match(entry, [r["ref_arxiv_id"] for r in refs])
    scored = [score_pair(entry, r, id_match=(r["ref_arxiv_id"] == matched_id)) for r in refs]
    scored.sort(key=lambda s: (-s["score"], s["ref_arxiv_id"]))
    return scored[:top_k]


def label_entry(
    entry: Dict[str, Any],
    refs: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> Tuple[Optional[str], float, str]:
    """Label one entry: (ref_arxiv_id or None, score, method).

    method is one of: arxiv_id, title_authors, no_match, no_candidates.
    """
    if not refs:
        return None, 0.0, "no_candidates"
    ranked = rank_candidates(entry, refs, top_k=2)
    best = ranked[0]
    if best["id_match"]:
        return best["ref_arxiv_id"], 1.0, "arxiv_id"
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    if best["score"] >= threshold and best["score"] - second_score >= margin:
        return best["ref_arxiv_id"], best["score"], "title_authors"
    return None, best["score"], "no_match"


def label_publication(
    arxiv_id: str,
    entries: List[Dict[str, Any]],
    refs: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> List[Dict[str, Any]]:
    """Label EVERY entry of a publication (None when no reference matches)."""
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        ref_id, score, method = label_entry(entry, refs, threshold, margin)
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "citation_key": entry.get("citation_key", ""),
                "groundtruth_arxiv_id": ref_id or "",
                "score": round(score, 4),
                "method": method,
                "source": "auto",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Corpus-level driver
# ---------------------------------------------------------------------------


def run_auto_labeling(
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manual_ids: Optional[Set[str]] = None,
    student_id: Optional[str] = None,
    fraction: float = 0.10,
    min_publications: int = 2,
    min_bibtex: int = 5,
    min_arxiv_refs: int = 1,
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
    seed: int = 42,
) -> Dict[str, Any]:
    """Automatically label a random sample of the publications outside the manual pool.

    The pool is every parsed publication except the manual ones. The number of
    publications to label is max(ceil(fraction * pool), min_publications), drawn at
    random (seeded) among publications with >= min_bibtex entries and >=
    min_arxiv_refs arXiv-keyed references.
    """
    manual = manual_ids or set()
    pool = [a for a in list_publications(intermediate_dir, student_id) if a not in manual]
    required = max(math.ceil(fraction * len(pool)), min_publications) if pool else 0

    stats = compute_publication_stats(pool, intermediate_dir, corpus_dir, show_progress=True)
    eligible = sorted(
        s["arxiv_id"]
        for s in stats
        if s["n_bibtex"] >= min_bibtex and s["n_arxiv_refs"] >= min_arxiv_refs
    )
    rng = random.Random(seed)
    selected = sorted(rng.sample(eligible, min(required, len(eligible))))

    rows: List[Dict[str, Any]] = []
    for aid in selected:
        rows.extend(
            label_publication(
                aid,
                load_entries(aid, intermediate_dir),
                load_references(aid, corpus_dir),
                threshold,
                margin,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / AUTO_LABELS_FILE).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / AUTO_PUBLICATIONS_FILE).write_text("\n".join(selected) + "\n", encoding="utf-8")

    matched = sum(1 for r in rows if r["groundtruth_arxiv_id"])
    return {
        "pool_size": len(pool),
        "required_publications": required,
        "eligible_publications": len(eligible),
        "selected_publications": len(selected),
        "entries_labelled": len(rows),
        "entries_matched": matched,
        "match_rate": round(matched / len(rows), 4) if rows else 0.0,
        "requirement_met": len(selected) >= required and len(selected) >= min_publications,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Automatic reference matcher / labeller")
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manual-file", type=Path, default=DEFAULT_OUTPUT_DIR / MANUAL_PUBLICATIONS_FILE)
    parser.add_argument("--fraction", type=float, default=0.10, help="Share of the pool to label (>= 0.10)")
    parser.add_argument("--min-publications", type=int, default=2, help="2*k required for the split")
    parser.add_argument("--min-bibtex", type=int, default=5)
    parser.add_argument("--min-arxiv-refs", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_auto_labeling(
        args.intermediate_dir,
        args.corpus_dir,
        args.output_dir,
        manual_ids=set(read_id_file(args.manual_file)),
        student_id=args.student_id,
        fraction=args.fraction,
        min_publications=args.min_publications,
        min_bibtex=args.min_bibtex,
        min_arxiv_refs=args.min_arxiv_refs,
        threshold=args.threshold,
        margin=args.margin,
        seed=args.seed,
    )
    for key, value in summary.items():
        print(f"{key:24s} {value}")
    if not summary["requirement_met"]:
        print(
            "WARNING: not enough eligible publications for the required share. "
            "Lower --min-bibtex / --min-arxiv-refs or check the parsing output."
        )


if __name__ == "__main__":
    main()