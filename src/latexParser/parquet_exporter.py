"""Parquet exporter module for Milestone 2.

Reads structured intermediate JSONs and exports three standardized Parquet tables:
1. elements.parquet
2. hierarchy.parquet
3. bibtex.parquet
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_INTERMEDIATE_DIR = Path("data/intermediate")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def export_parquet_tables(
    student_id: str,
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    arxiv_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export intermediate stores into three standardized Parquet tables.

    Args:
        student_id: Student identifier (e.g. '24127559').
        intermediate_dir: Path to directory containing parsed.json files.
        output_dir: Path where processed Parquet files will be stored.
        arxiv_ids: Optional list of arXiv IDs to restrict the export to.

    Returns:
        Summary dict containing counts and file paths of generated tables.
    """
    dest_dir = output_dir / student_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    elements_records: List[Dict[str, Any]] = []
    hierarchy_records: List[Dict[str, Any]] = []
    bibtex_records: List[Dict[str, Any]] = []

    # Determine publications to process
    if arxiv_ids is not None:
        target_dirs = [intermediate_dir / aid for aid in arxiv_ids]
    else:
        target_dirs = sorted(intermediate_dir.glob("*"))

    for p_dir in target_dirs:
        if not p_dir.is_dir():
            continue

        parsed_file = p_dir / "parsed.json"
        if not parsed_file.exists():
            continue

        try:
            with parsed_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            aid = data.get("arxiv_id", p_dir.name)

            for elem in data.get("elements", []):
                elements_records.append({
                    "element_id": str(elem.get("element_id", "")),
                    "arxiv_id": str(elem.get("arxiv_id", aid)),
                    "kind": str(elem.get("kind", "")),
                    "content": str(elem.get("content", "")),
                })

            for h in data.get("hierarchy", []):
                hierarchy_records.append({
                    "arxiv_id": str(h.get("arxiv_id", aid)),
                    "version": int(h.get("version", 1)),
                    "element_id": str(h.get("element_id", "")),
                    "parent_id": str(h.get("parent_id", "")),
                })

            for b in data.get("bibtex", []):
                authors_val = b.get("authors")
                if not isinstance(authors_val, list):
                    authors_val = [str(authors_val)] if authors_val else []

                year_val = b.get("year")
                try:
                    year_int = int(year_val) if year_val is not None else None
                except (ValueError, TypeError):
                    year_int = None

                bibtex_records.append({
                    "arxiv_id": str(b.get("arxiv_id", aid)),
                    "citation_key": str(b.get("citation_key", "")),
                    "entry_type": str(b.get("entry_type", "misc")),
                    "title": str(b.get("title") or "") if b.get("title") is not None else None,
                    "authors": authors_val,
                    "year": year_int,
                    "doi": str(b.get("doi")) if b.get("doi") is not None else None,
                    "journal": str(b.get("journal")) if b.get("journal") is not None else None,
                    "url": str(b.get("url")) if b.get("url") is not None else None,
                })
        except Exception:
            pass

    # Build DataFrames
    elements_df = pd.DataFrame(
        elements_records,
        columns=["element_id", "arxiv_id", "kind", "content"],
    )
    if not elements_df.empty:
        elements_df["kind"] = elements_df["kind"].astype("category")

    hierarchy_df = pd.DataFrame(
        hierarchy_records,
        columns=["arxiv_id", "version", "element_id", "parent_id"],
    )
    if not hierarchy_df.empty:
        hierarchy_df["version"] = hierarchy_df["version"].astype("int64")

    bibtex_df = pd.DataFrame(
        bibtex_records,
        columns=[
            "arxiv_id",
            "citation_key",
            "entry_type",
            "title",
            "authors",
            "year",
            "doi",
            "journal",
            "url",
        ],
    )
    if not bibtex_df.empty:
        bibtex_df["entry_type"] = bibtex_df["entry_type"].astype("category")
        bibtex_df["year"] = bibtex_df["year"].astype("Int64")

    elements_path = dest_dir / "elements.parquet"
    hierarchy_path = dest_dir / "hierarchy.parquet"
    bibtex_path = dest_dir / "bibtex.parquet"

    elements_df.to_parquet(elements_path, index=False, engine="pyarrow")
    hierarchy_df.to_parquet(hierarchy_path, index=False, engine="pyarrow")
    bibtex_df.to_parquet(bibtex_path, index=False, engine="pyarrow")

    return {
        "elements_count": len(elements_df),
        "hierarchy_count": len(hierarchy_df),
        "bibtex_count": len(bibtex_df),
        "elements_path": str(elements_path),
        "hierarchy_path": str(hierarchy_path),
        "bibtex_path": str(bibtex_path),
    }
