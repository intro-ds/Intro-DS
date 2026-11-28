from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_BIBTEX_PATH = PROCESSED_DIR / "bibtex.parquet"
DEFAULT_REFERENCES_PATH = PROCESSED_DIR / "references.parquet"
DEFAULT_SHEETS_DIR = PROCESSED_DIR / "labels" / "sheets"
DEFAULT_POOL_CANDIDATES = [
    PROCESSED_DIR / "labels" / "manual_pool.csv",
    PROCESSED_DIR / "labels" / "manual_pool.txt",
    ROOT / "config" / "manual_pool.csv",
    ROOT / "config" / "manual_pool.txt",
]

SHEET_COLUMNS = [
    "key",
    "title",
    "authors",
    "year",
    "doi",
    "groundtruth_arxiv_id",
    "cand_1_arxiv_id",
    "cand_1_title",
    "cand_1_score",
    "cand_2_arxiv_id",
    "cand_2_title",
    "cand_2_score",
    "cand_3_arxiv_id",
    "cand_3_title",
    "cand_3_score",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_title(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    tokens = _TOKEN_RE.findall(str(text).lower())
    return " ".join(tokens)


def simple_title_similarity(title_a: object, title_b: object) -> float:
    norm_a = _normalize_title(title_a)
    norm_b = _normalize_title(title_b)
    if not norm_a or not norm_b:
        return 0.0
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0
    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    return round(0.5 * jaccard + 0.5 * seq_ratio, 4)


def _format_authors(authors: object) -> str:
    if authors is None or (isinstance(authors, float) and pd.isna(authors)):
        return ""
    if isinstance(authors, (list, tuple)):
        return "; ".join(str(a).strip() for a in authors if str(a).strip())
    if hasattr(authors, "tolist"):
        return "; ".join(str(a).strip() for a in authors.tolist() if str(a).strip())
    return str(authors).strip()


def _format_year(year: object) -> str:
    if year is None or (isinstance(year, float) and pd.isna(year)):
        return ""
    try:
        return str(int(float(year)))
    except (TypeError, ValueError):
        return str(year).strip()


def _clean_str(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def rank_top_candidates(
    entry_title: object,
    ref_records: Sequence[dict],
    top_k: int = 3,
) -> List[tuple[str, str, float]]:
    scored: List[tuple[float, str, str]] = []
    for ref in ref_records:
        ref_id = _clean_str(ref.get("ref_arxiv_id"))
        if not ref_id:
            continue
        ref_title = _clean_str(ref.get("ref_title"))
        score = simple_title_similarity(entry_title, ref_title)
        scored.append((score, ref_id, ref_title))

    scored.sort(key=lambda item: (-item[0], item[1]))
    seen_ids: set[str] = set()
    top_items: List[tuple[str, str, float]] = []
    for score, ref_id, ref_title in scored:
        if ref_id in seen_ids:
            continue
        seen_ids.add(ref_id)
        top_items.append((ref_id, ref_title, score))
        if len(top_items) >= top_k:
            break
    return top_items


def export_label_sheet_for_paper(
    arxiv_id: str,
    bibtex_df: pd.DataFrame,
    references_df: pd.DataFrame,
    output_dir: Path = DEFAULT_SHEETS_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{arxiv_id}.csv"

    paper_bib = (
        bibtex_df[bibtex_df["arxiv_id"] == arxiv_id]
        if not bibtex_df.empty and "arxiv_id" in bibtex_df.columns
        else pd.DataFrame()
    )
    paper_refs = (
        references_df[references_df["arxiv_id"] == arxiv_id]
        if not references_df.empty and "arxiv_id" in references_df.columns
        else pd.DataFrame()
    )
    ref_records = paper_refs.to_dict(orient="records") if not paper_refs.empty else []

    rows: List[dict] = []
    for entry in paper_bib.to_dict(orient="records"):
        title = _clean_str(entry.get("title"))
        top_cands = rank_top_candidates(title, ref_records, top_k=3)
        row = {
            "key": _clean_str(entry.get("citation_key")),
            "title": title,
            "authors": _format_authors(entry.get("authors")),
            "year": _format_year(entry.get("year")),
            "doi": _clean_str(entry.get("doi")),
            "groundtruth_arxiv_id": "",
        }
        for idx in range(1, 4):
            if idx <= len(top_cands):
                cand_id, cand_title, cand_score = top_cands[idx - 1]
                row[f"cand_{idx}_arxiv_id"] = cand_id
                row[f"cand_{idx}_title"] = cand_title
                row[f"cand_{idx}_score"] = cand_score
            else:
                row[f"cand_{idx}_arxiv_id"] = ""
                row[f"cand_{idx}_title"] = ""
                row[f"cand_{idx}_score"] = ""
        rows.append(row)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def load_pool_ids(pool_path: Path) -> List[str]:
    pool_path = Path(pool_path)
    if not pool_path.exists():
        raise FileNotFoundError(f"Manual pool file not found: {pool_path}")

    ids: List[str] = []
    if pool_path.suffix.lower() == ".csv":
        with pool_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "arxiv_id" in reader.fieldnames:
                for row in reader:
                    val = (row.get("arxiv_id") or "").strip()
                    if val:
                        ids.append(val)
            else:
                f.seek(0)
                for row in csv.reader(f):
                    if row and row[0].strip() and row[0].strip() != "arxiv_id":
                        ids.append(row[0].strip())
    else:
        for line in pool_path.read_text(encoding="utf-8").splitlines():
            val = line.strip()
            if val and not val.startswith("#"):
                ids.append(val)

    return list(dict.fromkeys(ids))


def generate_label_sheets(
    arxiv_ids: Iterable[str],
    bibtex_path: Path = DEFAULT_BIBTEX_PATH,
    references_path: Path = DEFAULT_REFERENCES_PATH,
    output_dir: Path = DEFAULT_SHEETS_DIR,
) -> List[Path]:
    bibtex_df = pd.read_parquet(bibtex_path) if Path(bibtex_path).exists() else pd.DataFrame()
    references_df = (
        pd.read_parquet(references_path) if Path(references_path).exists() else pd.DataFrame()
    )
    written: List[Path] = []
    for aid in arxiv_ids:
        aid = str(aid).strip()
        if not aid:
            continue
        written.append(
            export_label_sheet_for_paper(
                arxiv_id=aid,
                bibtex_df=bibtex_df,
                references_df=references_df,
                output_dir=output_dir,
            )
        )
    return written


def _resolve_target_ids(
    arxiv_ids: Optional[Sequence[str]],
    pool_file: Optional[Path],
    bibtex_path: Path,
) -> List[str]:
    if arxiv_ids:
        return list(dict.fromkeys(aid.strip() for aid in arxiv_ids if aid.strip()))
    if pool_file is not None:
        return load_pool_ids(pool_file)
    for candidate in DEFAULT_POOL_CANDIDATES:
        if candidate.exists():
            return load_pool_ids(candidate)
    if Path(bibtex_path).exists():
        df = pd.read_parquet(bibtex_path, columns=["arxiv_id"])
        counts = df["arxiv_id"].value_counts()
        eligible = counts[counts >= 20].index.tolist()
        return eligible[:5] if eligible else counts.index.tolist()[:5]
    return []


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate manual labeling CSV sheets with top-3 suggested candidates."
    )
    ap.add_argument("--arxiv-ids", nargs="*", help="Specific arXiv IDs in the manual pool")
    ap.add_argument("--pool-file", type=Path, default=None, help="Path to manual pool CSV/TXT")
    ap.add_argument("--bibtex", type=Path, default=DEFAULT_BIBTEX_PATH)
    ap.add_argument("--references", type=Path, default=DEFAULT_REFERENCES_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_SHEETS_DIR)
    args = ap.parse_args(argv)

    target_ids = _resolve_target_ids(args.arxiv_ids, args.pool_file, args.bibtex)
    if not target_ids:
        print("No target publications found for label sheet generation.")
        return 0

    paths = generate_label_sheets(
        arxiv_ids=target_ids,
        bibtex_path=args.bibtex,
        references_path=args.references,
        output_dir=args.output_dir,
    )
    print(f"Generated {len(paths)} label sheet(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())