"""Pair features for the reference-matching ranker (Milestone 3).

The task is framed as learning to rank: for every BibTeX entry, the candidates are all
arXiv-keyed references of the SAME publication, and one (entry, reference) pair is one
row. Features reuse the normalisation helpers of auto_labeler so that both pipelines
see the same text.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.matching import auto_labeler as al

NAN = float("nan")

BASE_FEATURES = [
    "title_jaccard",       # token overlap / union, robust to word order
    "title_containment",   # overlap / smaller title: catches truncated titles
    "title_seq",           # character-level ratio: catches typos and LaTeX residue
    "title_sim",           # title similarity used by the rule-based labeler
    "author_sim",          # last-name overlap
    "first_author_match",  # 1 when the first authors agree
    "n_common_authors",
    "year_diff",           # |entry year - reference year|
    "year_sim",
    "id_match",            # entry text mentions the reference arXiv ID
    "heuristic_score",     # weighted score of the rule-based labeler
    "entry_has_title",
    "entry_n_authors",
    "entry_has_year",
    "entry_has_doi",
    "entry_has_url",
    "entry_title_len",
    "ref_title_len",
    "title_len_ratio",
    "n_refs",
]
# Features computed relative to the other candidates of the same entry / reference
RELATIVE_FEATURES = [
    "score_rank",      # rank of the reference among all references for this entry
    "score_gap",       # best score of the entry minus this score
    "score_rank_rev",  # rank of the entry among all entries for this reference
    "title_seq_gap",
]
FEATURE_COLUMNS = BASE_FEATURES + RELATIVE_FEATURES

# Features derived from the rule-based labeler that produced the automatic labels.
# Used for the leakage ablation (train_ranker --drop-leaky).
LEAKY_FEATURES = ["id_match", "heuristic_score", "score_rank", "score_gap", "score_rank_rev"]

_YEAR_RE = re.compile(r"(\d{4})")


def _num(value: Optional[float]) -> float:
    """Unknown (None) becomes NaN; LightGBM handles NaN natively."""
    return NAN if value is None else float(value)


def _ref_year(date: Any) -> Optional[int]:
    match = _YEAR_RE.match(str(date or ""))
    return int(match.group(1)) if match else None


def _entry_year(year: Any) -> Optional[int]:
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def publication_features(entries: List[Dict[str, Any]], refs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build the m x n pair-feature table of one publication.

    Rows are ordered entry-major (row = i * n + j for entry i and reference j), which the
    callers rely on to reshape per-pair arrays into (m, n) matrices.

    Returns:
        DataFrame with columns entry_idx, citation_key, ref_arxiv_id and FEATURE_COLUMNS.
    """
    m, n = len(entries), len(refs)
    columns = ["entry_idx", "citation_key", "ref_arxiv_id"] + FEATURE_COLUMNS
    if m == 0 or n == 0:
        return pd.DataFrame(columns=columns)

    ref_ids = [r["ref_arxiv_id"] for r in refs]
    ref_tokens = [al._title_tokens(r.get("ref_title")) for r in refs]
    ref_norm = [al.normalize_text(r.get("ref_title")) for r in refs]
    ref_last = [al._last_names(tuple(r.get("ref_authors") or ())) for r in refs]
    ref_year = [_ref_year(r.get("ref_submission_date")) for r in refs]

    records: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        e_tokens = al._title_tokens(entry.get("title"))
        e_norm = al.normalize_text(entry.get("title"))
        e_last = al._last_names(tuple(entry.get("authors") or ()))
        e_year = _entry_year(entry.get("year"))
        matched_id = al.find_arxiv_id_match(entry, ref_ids)

        for j, ref in enumerate(refs):
            r_tokens = ref_tokens[j]
            both = bool(e_tokens) and bool(r_tokens)
            inter = len(e_tokens & r_tokens)
            jaccard = containment = seq = ratio = NAN
            if both:
                jaccard = inter / len(e_tokens | r_tokens)
                containment = inter / min(len(e_tokens), len(r_tokens))
                # Character ratio only for plausible pairs (speed), as in auto_labeler
                seq = SequenceMatcher(None, e_norm, ref_norm[j]).ratio() if jaccard >= 0.1 else 0.0
                ratio = min(len(e_tokens), len(r_tokens)) / max(len(e_tokens), len(r_tokens))

            title_sim = al.title_similarity(entry.get("title"), ref.get("ref_title"))
            author_sim = al.author_similarity(entry.get("authors"), ref.get("ref_authors"))
            year_sim = al.year_similarity(entry.get("year"), ref.get("ref_submission_date"))
            id_match = ref_ids[j] == matched_id
            heuristic = 1.0 if id_match else al.combine_scores(title_sim, author_sim, year_sim)

            both_authors = bool(e_last) and bool(ref_last[j])
            records.append(
                {
                    "entry_idx": i,
                    "citation_key": entry.get("citation_key", ""),
                    "ref_arxiv_id": ref_ids[j],
                    "title_jaccard": jaccard,
                    "title_containment": containment,
                    "title_seq": seq,
                    "title_sim": _num(title_sim),
                    "author_sim": _num(author_sim),
                    "first_author_match": float(e_last[0] == ref_last[j][0]) if both_authors else NAN,
                    "n_common_authors": float(len(set(e_last) & set(ref_last[j]))),
                    "year_diff": float(abs(e_year - ref_year[j])) if (e_year and ref_year[j]) else NAN,
                    "year_sim": _num(year_sim),
                    "id_match": float(id_match),
                    "heuristic_score": float(heuristic),
                    "entry_has_title": float(bool((entry.get("title") or "").strip())),
                    "entry_n_authors": float(len(entry.get("authors") or [])),
                    "entry_has_year": float(e_year is not None),
                    "entry_has_doi": float(bool(entry.get("doi"))),
                    "entry_has_url": float(bool(entry.get("url"))),
                    "entry_title_len": float(len(e_tokens)),
                    "ref_title_len": float(len(r_tokens)),
                    "title_len_ratio": ratio,
                    "n_refs": float(n),
                }
            )

    df = pd.DataFrame.from_records(records)

    # Relative features: reshape the entry-major rows into (entries x references) matrices
    h = df["heuristic_score"].to_numpy(dtype=float).reshape(m, n)
    h_frame = pd.DataFrame(h)
    df["score_rank"] = h_frame.rank(axis=1, ascending=False, method="min").to_numpy().ravel()
    df["score_rank_rev"] = h_frame.rank(axis=0, ascending=False, method="min").to_numpy().ravel()
    df["score_gap"] = (h.max(axis=1, keepdims=True) - h).ravel()
    seq_matrix = np.nan_to_num(df["title_seq"].to_numpy(dtype=float), nan=0.0).reshape(m, n)
    df["title_seq_gap"] = (seq_matrix.max(axis=1, keepdims=True) - seq_matrix).ravel()

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype("float32")
    return df[columns]