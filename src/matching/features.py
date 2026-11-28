from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_BIBTEX_PATH = PROCESSED_DIR / "bibtex.parquet"
DEFAULT_REFERENCES_PATH = PROCESSED_DIR / "references.parquet"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ARXIV_ID_RE = re.compile(r"(?:arxiv[:/\s]*)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})", re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)

FEATURE_COLUMNS = [
    "arxiv_id",
    "citation_key",
    "ref_arxiv_id",
    "title_jaccard",
    "title_char_tfidf_cosine",
    "title_fuzz_ratio",
    "title_fuzz_token_set",
    "title_len_diff",
    "title_len_ratio",
    "first_author_last_name_match",
    "author_last_names_jaccard",
    "year_diff",
    "year_exact_match",
    "doi_match",
    "arxiv_id_match",
    "title_jaccard_rank",
    "title_jaccard_diff_from_best",
    "title_char_tfidf_cosine_rank",
    "title_char_tfidf_cosine_diff_from_best",
    "title_fuzz_token_set_rank",
    "title_fuzz_token_set_diff_from_best",
]


def _is_missing(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


def _clean_text(val: object) -> str:
    if _is_missing(val):
        return ""
    return " ".join(_TOKEN_RE.findall(str(val).lower()))


def _extract_author_list(authors: object) -> List[str]:
    if _is_missing(authors):
        return []
    if isinstance(authors, str):
        parts = re.split(r"\s+and\s+|;", authors)
        return [p.strip() for p in parts if p.strip()]
    if hasattr(authors, "tolist"):
        return [str(a).strip() for a in authors.tolist() if str(a).strip()]
    if isinstance(authors, Iterable):
        return [str(a).strip() for a in authors if str(a).strip()]
    return []


def _last_name(author_name: str) -> str:
    name = author_name.strip()
    if not name:
        return ""
    if "," in name:
        surname_part = name.split(",", 1)[0]
    else:
        tokens = name.split()
        surname_part = tokens[-1] if tokens else ""
    cleaned = _TOKEN_RE.findall(surname_part.lower())
    return cleaned[-1] if cleaned else ""


def _extract_last_names(authors: object) -> List[str]:
    names = [_last_name(a) for a in _extract_author_list(authors)]
    return [n for n in names if n]


def _extract_year(val: object) -> Optional[int]:
    if _is_missing(val):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return int(val)
    match = re.search(r"(19|20)\d{2}", str(val))
    return int(match.group(0)) if match else None


def _normalize_doi(val: object) -> str:
    if _is_missing(val):
        return ""
    s = _DOI_PREFIX_RE.sub("", str(val).strip().lower())
    return s.strip()


def _extract_arxiv_ids_from_entry(entry: dict) -> Set[str]:
    found: Set[str] = set()
    for field in ("url", "doi", "journal", "title", "citation_key"):
        val = entry.get(field)
        if _is_missing(val):
            continue
        for match in _ARXIV_ID_RE.findall(str(val).lower()):
            found.add(match.strip())
    return found


def _normalize_arxiv_id(val: object) -> str:
    if _is_missing(val):
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"v\d+$", "", s)
    return s


def _compute_char_tfidf_matrix(
    entry_titles: Sequence[str],
    ref_titles: Sequence[str],
) -> np.ndarray:
    n_entries = len(entry_titles)
    n_refs = len(ref_titles)
    if n_entries == 0 or n_refs == 0:
        return np.zeros((n_entries, n_refs), dtype=float)

    corpus = list(entry_titles) + list(ref_titles)
    if not any(t.strip() for t in corpus):
        return np.zeros((n_entries, n_refs), dtype=float)

    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        tfidf = vec.fit_transform(corpus)
        sim = linear_kernel(tfidf[:n_entries], tfidf[n_entries:])
        return np.asarray(sim, dtype=float)
    except ValueError:
        return np.zeros((n_entries, n_refs), dtype=float)


def build_pair_features_from_frames(
    arxiv_id: str,
    bibtex_df: pd.DataFrame,
    references_df: pd.DataFrame,
) -> pd.DataFrame:
    paper_bib = (
        bibtex_df[bibtex_df["arxiv_id"] == arxiv_id].reset_index(drop=True)
        if not bibtex_df.empty and "arxiv_id" in bibtex_df.columns
        else pd.DataFrame()
    )
    paper_refs = (
        references_df[references_df["arxiv_id"] == arxiv_id].reset_index(drop=True)
        if not references_df.empty and "arxiv_id" in references_df.columns
        else pd.DataFrame()
    )

    if paper_bib.empty or paper_refs.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    bib_records = paper_bib.to_dict(orient="records")
    ref_records = paper_refs.to_dict(orient="records")

    entry_norm_titles = [_clean_text(r.get("title")) for r in bib_records]
    ref_norm_titles = [_clean_text(r.get("ref_title")) for r in ref_records]
    tfidf_sim = _compute_char_tfidf_matrix(entry_norm_titles, ref_norm_titles)

    entry_token_sets = [set(t.split()) for t in entry_norm_titles]
    ref_token_sets = [set(t.split()) for t in ref_norm_titles]

    entry_authors = [_extract_last_names(r.get("authors")) for r in bib_records]
    ref_authors = [_extract_last_names(r.get("ref_authors")) for r in ref_records]

    entry_years = [_extract_year(r.get("year")) for r in bib_records]
    ref_years = [_extract_year(r.get("ref_submission_date")) for r in ref_records]

    entry_dois = [_normalize_doi(r.get("doi")) for r in bib_records]
    entry_arxiv_candidates = [_extract_arxiv_ids_from_entry(r) for r in bib_records]
    ref_arxiv_ids = [_normalize_arxiv_id(r.get("ref_arxiv_id")) for r in ref_records]

    rows: List[dict] = []
    for i, b_rec in enumerate(bib_records):
        e_title = entry_norm_titles[i]
        e_tokens = entry_token_sets[i]
        e_len = len(e_title)
        e_auth = entry_authors[i]
        e_auth_set = set(e_auth)
        e_first_last = e_auth[0] if e_auth else ""
        e_year = entry_years[i]
        e_doi = entry_dois[i]
        e_extracted_ids = entry_arxiv_candidates[i]

        for j, r_rec in enumerate(ref_records):
            r_id_raw = str(r_rec.get("ref_arxiv_id") or "").strip()
            r_id_norm = ref_arxiv_ids[j]
            r_title = ref_norm_titles[j]
            r_tokens = ref_token_sets[j]
            r_len = len(r_title)
            r_auth = ref_authors[j]
            r_auth_set = set(r_auth)
            r_first_last = r_auth[0] if r_auth else ""
            r_year = ref_years[j]

            union_tokens = e_tokens | r_tokens
            title_jaccard = (
                len(e_tokens & r_tokens) / len(union_tokens) if union_tokens else 0.0
            )
            title_tfidf = float(tfidf_sim[i, j])
            if e_title and r_title:
                title_fuzz_ratio = float(fuzz.ratio(e_title, r_title)) / 100.0
                title_fuzz_token_set = float(fuzz.token_set_ratio(e_title, r_title)) / 100.0
            else:
                title_fuzz_ratio = 0.0
                title_fuzz_token_set = 0.0

            title_len_diff = abs(e_len - r_len)
            max_len = max(e_len, r_len)
            title_len_ratio = (min(e_len, r_len) / max_len) if max_len > 0 else 0.0

            first_author_match = int(
                bool(e_first_last and r_first_last and e_first_last == r_first_last)
            )
            union_authors = e_auth_set | r_auth_set
            author_jaccard = (
                len(e_auth_set & r_auth_set) / len(union_authors) if union_authors else 0.0
            )

            if e_year is not None and r_year is not None:
                year_diff = float(abs(e_year - r_year))
                year_exact = int(e_year == r_year)
            else:
                year_diff = np.nan
                year_exact = 0

            doi_match = int(
                bool(e_doi and r_id_norm and f"arxiv.{r_id_norm}" in e_doi)
            )
            arxiv_id_match = int(
                bool(r_id_norm and r_id_norm in e_extracted_ids)
            )

            rows.append({
                "arxiv_id": arxiv_id,
                "citation_key": str(b_rec.get("citation_key") or ""),
                "ref_arxiv_id": r_id_raw,
                "title_jaccard": round(title_jaccard, 6),
                "title_char_tfidf_cosine": round(title_tfidf, 6),
                "title_fuzz_ratio": round(title_fuzz_ratio, 6),
                "title_fuzz_token_set": round(title_fuzz_token_set, 6),
                "title_len_diff": int(title_len_diff),
                "title_len_ratio": round(title_len_ratio, 6),
                "first_author_last_name_match": first_author_match,
                "author_last_names_jaccard": round(author_jaccard, 6),
                "year_diff": year_diff,
                "year_exact_match": year_exact,
                "doi_match": doi_match,
                "arxiv_id_match": arxiv_id_match,
            })

    df = pd.DataFrame(rows)
    for col in ("title_jaccard", "title_char_tfidf_cosine", "title_fuzz_token_set"):
        grp = df.groupby("citation_key", sort=False)[col]
        df[f"{col}_rank"] = grp.rank(ascending=False, method="min").astype(int)
        df[f"{col}_diff_from_best"] = (grp.transform("max") - df[col]).round(6)

    return df[FEATURE_COLUMNS]


def build_pair_features(
    arxiv_id: str,
    bibtex_path: Path = DEFAULT_BIBTEX_PATH,
    references_path: Path = DEFAULT_REFERENCES_PATH,
) -> pd.DataFrame:
    bibtex_df = pd.read_parquet(bibtex_path) if Path(bibtex_path).exists() else pd.DataFrame()
    references_df = (
        pd.read_parquet(references_path) if Path(references_path).exists() else pd.DataFrame()
    )
    return build_pair_features_from_frames(arxiv_id, bibtex_df, references_df)