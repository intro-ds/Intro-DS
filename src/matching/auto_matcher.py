"""Automatically label Semantic Scholar references against the local paper corpus."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[2]
PAPERS_PATH = ROOT / "data" / "processed" / "papers.parquet"
METADATA_PATH = ROOT / "data" / "raw" / "arxiv_metadata_all_assignments.jsonl"
SEMANTIC_SCHOLAR_DIR = ROOT / "data" / "raw" / "semantic_scholar"
_DEFAULT_METADATA_PATH = ROOT / "data" / "raw" / "arxiv_metadata_all_assignments.jsonl"

FUZZY_TITLE_THRESHOLD = 90.0
MATCHED = "matched"
NO_MATCH = "no_match"

_ARXIV_ID = re.compile(
    r"(?<![\w.])(?:arxiv\s*:\s*)?"
    r"(?P<identifier>(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})v?\d*)"
    r"(?![\w.])",
    re.IGNORECASE,
)
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_DOI_PREFIX = re.compile(
    r"^(?:(?:https?://)?(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE
)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\*?")


def _text(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalize_arxiv_id(value: object) -> Optional[str]:
    text = _text(value)
    if not text:
        return None
    match = _ARXIV_ID.search(text)
    if not match:
        return None
    return _ARXIV_VERSION.sub("", match.group("identifier")).casefold()


def _normalize_doi(value: object) -> str:
    doi = _DOI_PREFIX.sub("", _text(value)).rstrip(".,; ").casefold()
    return doi


def _normalize_title(value: object) -> str:
    title = unicodedata.normalize("NFKC", _text(value))
    title = _LATEX_COMMAND.sub(" ", title)
    title = unicodedata.normalize("NFKD", title).casefold()
    title = "".join(char for char in title if not unicodedata.combining(char))
    return " ".join(re.sub(r"[\W_]+", " ", title, flags=re.UNICODE).split())


def _year(value: object) -> str:
    match = re.search(r"\b(?:19|20)\d{2}\b", _text(value))
    return match.group(0) if match else ""


def _author_name(author: object) -> str:
    if isinstance(author, Mapping):
        return _text(author.get("name"))
    return _text(author)


def _authors_sequence(authors: object) -> List[object]:
    if authors is None:
        return []
    if isinstance(authors, str):
        return [authors]
    if isinstance(authors, Mapping):
        return []
    try:
        return list(authors)
    except TypeError:
        return []


def _first_author_surname(authors: object) -> str:
    author_list = _authors_sequence(authors)
    if not author_list:
        return ""
    name = _author_name(author_list[0])
    if not name:
        return ""
    surname = name.split(",", 1)[0] if "," in name else name.split()[-1]
    normalized = _normalize_title(surname)
    return normalized.split()[-1] if normalized else ""


def _external_ids(reference: Mapping[str, object]) -> Mapping[str, object]:
    external_ids = reference.get("externalIds")
    if external_ids is None:
        return {}
    if not isinstance(external_ids, Mapping):
        raise ValueError("Semantic Scholar externalIds must be an object")
    return external_ids


def _candidate_metadata_paths() -> Tuple[Path, ...]:
    metadata_path = Path(METADATA_PATH)
    paths = [metadata_path]
    if metadata_path == _DEFAULT_METADATA_PATH:
        paths.extend(
            path
            for path in sorted((ROOT / "data" / "raw").glob(
                "*/arxiv_metadata_*_all_versions.jsonl"
            ))
            if path.stat().st_size
        )
    return tuple(dict.fromkeys(paths))


def _metadata_dois(metadata_paths: Sequence[Path]) -> Dict[str, set[str]]:
    dois_by_id: Dict[str, set[str]] = {}
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {metadata_path} at line {line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Invalid metadata record in {metadata_path} "
                        f"at line {line_number}"
                    )
                arxiv_id = _normalize_arxiv_id(record.get("id"))
                doi = _normalize_doi(record.get("doi"))
                if arxiv_id and doi:
                    dois_by_id.setdefault(arxiv_id, set()).add(doi)
    return dois_by_id


@lru_cache(maxsize=4)
def _load_candidates(
    papers_path_text: str,
    papers_mtime_ns: int,
    papers_size: int,
    metadata_signatures: Tuple[Tuple[str, int, int], ...],
) -> Tuple[dict, ...]:
    del papers_mtime_ns, papers_size
    papers_path = Path(papers_path_text)
    metadata_paths = tuple(Path(signature[0]) for signature in metadata_signatures)
    frame = pd.read_parquet(papers_path)
    required = {"arxiv_id", "title", "authors", "submission_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{papers_path} is missing required columns: {', '.join(sorted(missing))}"
        )

    metadata_dois = _metadata_dois(metadata_paths)
    candidates = []
    seen_ids = set()
    for row in frame.to_dict(orient="records"):
        arxiv_id = _normalize_arxiv_id(row.get("arxiv_id"))
        if not arxiv_id:
            raise ValueError(
                f"Invalid arXiv ID in {papers_path}: {row.get('arxiv_id')!r}"
            )
        if arxiv_id in seen_ids:
            raise ValueError(f"Duplicate arXiv ID in {papers_path}: {arxiv_id}")
        seen_ids.add(arxiv_id)

        dois = set(metadata_dois.get(arxiv_id, set()))
        dois.add(f"10.48550/arxiv.{arxiv_id}")
        candidates.append({
            "arxiv_id": arxiv_id,
            "title": _text(row.get("title")),
            "normalized_title": _normalize_title(row.get("title")),
            "first_author_surname": _first_author_surname(row.get("authors")),
            "year": _year(row.get("submission_date")),
            "dois": dois,
        })
    return tuple(candidates)


def _unique_id(matches: Sequence[dict]) -> Optional[str]:
    identifiers = {candidate["arxiv_id"] for candidate in matches}
    return next(iter(identifiers)) if len(identifiers) == 1 else None


def _match_reference(
    reference: Mapping[str, object],
    candidates: Sequence[dict],
    id_index: Mapping[str, dict],
    doi_index: Mapping[str, Sequence[dict]],
    title_index: Mapping[str, Sequence[dict]],
) -> Tuple[Optional[str], Optional[str]]:
    external_ids = _external_ids(reference)

    ref_arxiv_id = _normalize_arxiv_id(external_ids.get("ArXiv"))
    if ref_arxiv_id in id_index:
        return ref_arxiv_id, "arxiv_id"

    doi = _normalize_doi(external_ids.get("DOI"))
    if doi:
        match = _unique_id(doi_index.get(doi, ()))
        if match:
            return match, "doi"
        arxiv_doi = _ARXIV_DOI.fullmatch(doi)
        doi_arxiv_id = _normalize_arxiv_id(arxiv_doi.group(1)) if arxiv_doi else None
        if doi_arxiv_id in id_index:
            return doi_arxiv_id, "doi"

    normalized_title = _normalize_title(reference.get("title"))
    if not normalized_title:
        return None, None
    exact_match = _unique_id(title_index.get(normalized_title, ()))
    if exact_match:
        return exact_match, "title"

    authors = reference.get("authors") or []
    first_author_surname = _first_author_surname(authors)
    reference_year = _year(
        reference.get("year") or reference.get("publicationDate")
    )
    if not first_author_surname or not reference_year:
        return None, None

    compatible = [
        candidate
        for candidate in candidates
        if candidate["year"] == reference_year
        and candidate["first_author_surname"] == first_author_surname
        and candidate["normalized_title"]
    ]
    scored = [
        (
            float(fuzz.ratio(normalized_title, candidate["normalized_title"])),
            candidate["arxiv_id"],
        )
        for candidate in compatible
    ]
    if not scored:
        return None, None
    best_score = max(score for score, _ in scored)
    best_ids = {identifier for score, identifier in scored if score == best_score}
    if best_score < FUZZY_TITLE_THRESHOLD or len(best_ids) != 1:
        return None, None
    return next(iter(best_ids)), "fuzzy_title"


def _label_references(
    arxiv_id: str,
    references: object,
    candidates: Sequence[dict],
) -> List[dict]:
    if references is None:
        references = []
    if not isinstance(references, list):
        raise ValueError("Semantic Scholar references must be a list")
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            raise ValueError(f"Invalid reference at index {index}: expected an object")

    id_index = {candidate["arxiv_id"]: candidate for candidate in candidates}
    doi_index: Dict[str, List[dict]] = {}
    title_index: Dict[str, List[dict]] = {}
    for candidate in candidates:
        for doi in candidate["dois"]:
            doi_index.setdefault(doi, []).append(candidate)
        if candidate["normalized_title"]:
            title_index.setdefault(candidate["normalized_title"], []).append(candidate)

    labelled = []
    for index, reference in enumerate(references):
        matched_arxiv_id, method = _match_reference(
            reference, candidates, id_index, doi_index, title_index
        )
        external_ids = _external_ids(reference)
        authors = _authors_sequence(reference.get("authors"))
        labelled.append({
            "arxiv_id": arxiv_id,
            "reference_index": index,
            "ref_arxiv_id": _normalize_arxiv_id(external_ids.get("ArXiv")),
            "ref_doi": _normalize_doi(external_ids.get("DOI")) or None,
            "ref_title": _text(reference.get("title")) or None,
            "ref_authors": [
                name for author in authors if (name := _author_name(author))
            ],
            "ref_year": _year(
                reference.get("year") or reference.get("publicationDate")
            ) or None,
            "ref_submission_date": _text(reference.get("publicationDate")) or None,
            "ref_s2_id": _text(reference.get("paperId")) or None,
            "label": MATCHED if matched_arxiv_id else NO_MATCH,
            "matched_arxiv_id": matched_arxiv_id,
            "match_method": method,
        })
    return labelled


def auto_label(arxiv_id: str) -> List[dict]:
    """Label every Semantic Scholar reference for ``arxiv_id``.

    Matches are attempted by arXiv ID/DOI, normalized title, then fuzzy title
    with the same first-author surname and publication year. The returned list
    has one row per source reference, including unmatched entries.
    """
    raw_arxiv_id = _text(arxiv_id)
    if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", raw_arxiv_id, re.IGNORECASE):
        raise ValueError(f"Invalid arXiv ID: {arxiv_id!r}")
    normalized_arxiv_id = _normalize_arxiv_id(raw_arxiv_id)
    if not normalized_arxiv_id:
        raise ValueError(f"Invalid arXiv ID: {arxiv_id!r}")

    source_path = SEMANTIC_SCHOLAR_DIR / f"{normalized_arxiv_id}.json"
    with source_path.open("r", encoding="utf-8") as file:
        try:
            payload = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Semantic Scholar JSON: {source_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Semantic Scholar record: {source_path}")
    record_id = _normalize_arxiv_id(payload.get("arxiv_id"))
    if record_id and record_id != normalized_arxiv_id:
        raise ValueError(
            f"Semantic Scholar record ID does not match {normalized_arxiv_id}"
        )

    papers_path = Path(PAPERS_PATH)
    papers_stat = papers_path.stat()
    metadata_signatures = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in _candidate_metadata_paths()
    )
    candidates = _load_candidates(
        str(papers_path),
        papers_stat.st_mtime_ns,
        papers_stat.st_size,
        metadata_signatures,
    )
    return _label_references(normalized_arxiv_id, payload.get("references"), candidates)
