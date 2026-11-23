"""Parquet exporter module for Milestone 2.

Reads structured intermediate JSONs and exports three standardized Parquet tables:
1. elements.parquet
2. hierarchy.parquet
3. bibtex.parquet

Rows are streamed to disk in batches with pyarrow.ParquetWriter so that memory
stays bounded regardless of corpus size.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_INTERMEDIATE_DIR = Path("data/intermediate")
DEFAULT_PROCESSED_DIR = Path("data/processed")

# Number of buffered rows (per table) that triggers a flush to disk.
FLUSH_ROWS = 100_000

_KIND_TYPE = pa.dictionary(pa.int32(), pa.string())

ELEMENTS_SCHEMA = pa.schema([
    ("element_id", pa.string()),
    ("arxiv_id", pa.string()),
    ("kind", _KIND_TYPE),
    ("content", pa.string()),
])

HIERARCHY_SCHEMA = pa.schema([
    ("arxiv_id", pa.string()),
    ("version", pa.int64()),
    ("element_id", pa.string()),
    ("parent_id", pa.string()),
])

BIBTEX_SCHEMA = pa.schema([
    ("arxiv_id", pa.string()),
    ("citation_key", pa.string()),
    ("entry_type", _KIND_TYPE),
    ("title", pa.string()),
    ("authors", pa.list_(pa.string())),
    ("year", pa.int32()),
    ("doi", pa.string()),
    ("journal", pa.string()),
    ("url", pa.string()),
])


def _to_table(records: List[Dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Convert a list of row dicts into an Arrow table with the given schema."""
    columns = {
        field.name: pa.array([r[field.name] for r in records], type=field.type)
        for field in schema
    }
    return pa.table(columns, schema=schema)


def _flush(writer: pq.ParquetWriter, records: List[Dict[str, Any]], schema: pa.Schema) -> None:
    """Write buffered rows as one row group and clear the buffer."""
    if records:
        writer.write_table(_to_table(records, schema))
        records.clear()


def export_parquet_tables(
    student_id: str,
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    arxiv_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export intermediate stores into three standardized Parquet tables.

    Args:
        student_id: Student identifier (e.g. '24127590').
        intermediate_dir: Path to directory containing parsed.json files.
        output_dir: Path where processed Parquet files will be stored.
        arxiv_ids: Optional list of arXiv IDs to restrict the export to.

    Returns:
        Summary dict containing counts and file paths of generated tables.
    """
    dest_dir = output_dir / student_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    elements_path = dest_dir / "elements.parquet"
    hierarchy_path = dest_dir / "hierarchy.parquet"
    bibtex_path = dest_dir / "bibtex.parquet"

    if arxiv_ids is not None:
        target_dirs = [intermediate_dir / aid for aid in arxiv_ids]
    else:
        target_dirs = sorted(intermediate_dir.glob("*"))

    elements_buf: List[Dict[str, Any]] = []
    hierarchy_buf: List[Dict[str, Any]] = []
    bibtex_buf: List[Dict[str, Any]] = []
    counts = {"elements": 0, "hierarchy": 0, "bibtex": 0}
    skipped: List[str] = []

    with pq.ParquetWriter(elements_path, ELEMENTS_SCHEMA, compression="zstd") as w_elem, \
            pq.ParquetWriter(hierarchy_path, HIERARCHY_SCHEMA, compression="zstd") as w_hier, \
            pq.ParquetWriter(bibtex_path, BIBTEX_SCHEMA, compression="zstd") as w_bib:

        for p_dir in target_dirs:
            parsed_file = p_dir / "parsed.json"
            if not p_dir.is_dir() or not parsed_file.exists():
                continue

            try:
                with parsed_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                skipped.append(p_dir.name)
                continue

            aid = data.get("arxiv_id", p_dir.name)

            for elem in data.get("elements", []):
                elements_buf.append({
                    "element_id": str(elem.get("element_id", "")),
                    "arxiv_id": str(elem.get("arxiv_id", aid)),
                    "kind": str(elem.get("kind", "")),
                    "content": str(elem.get("content", "")),
                })

            for h in data.get("hierarchy", []):
                hierarchy_buf.append({
                    "arxiv_id": str(h.get("arxiv_id", aid)),
                    "version": int(h.get("version", 1)),
                    "element_id": str(h.get("element_id", "")),
                    "parent_id": str(h.get("parent_id", "")),
                })

            for b in data.get("bibtex", []):
                authors_val = b.get("authors")
                if not isinstance(authors_val, list):
                    authors_val = [str(authors_val)] if authors_val else []
                authors_val = [str(a) for a in authors_val]

                try:
                    year_val = b.get("year")
                    year_int = int(year_val) if year_val is not None else None
                except (ValueError, TypeError):
                    year_int = None

                def _opt(key: str) -> Optional[str]:
                    val = b.get(key)
                    return str(val) if val is not None else None

                bibtex_buf.append({
                    "arxiv_id": str(b.get("arxiv_id", aid)),
                    "citation_key": str(b.get("citation_key", "")),
                    "entry_type": str(b.get("entry_type", "misc")),
                    "title": _opt("title"),
                    "authors": authors_val,
                    "year": year_int,
                    "doi": _opt("doi"),
                    "journal": _opt("journal"),
                    "url": _opt("url"),
                })

            # Flush each table independently once its buffer is large enough
            if len(elements_buf) >= FLUSH_ROWS:
                counts["elements"] += len(elements_buf)
                _flush(w_elem, elements_buf, ELEMENTS_SCHEMA)
            if len(hierarchy_buf) >= FLUSH_ROWS:
                counts["hierarchy"] += len(hierarchy_buf)
                _flush(w_hier, hierarchy_buf, HIERARCHY_SCHEMA)
            if len(bibtex_buf) >= FLUSH_ROWS:
                counts["bibtex"] += len(bibtex_buf)
                _flush(w_bib, bibtex_buf, BIBTEX_SCHEMA)

        # Final flush of remaining rows
        counts["elements"] += len(elements_buf)
        counts["hierarchy"] += len(hierarchy_buf)
        counts["bibtex"] += len(bibtex_buf)
        _flush(w_elem, elements_buf, ELEMENTS_SCHEMA)
        _flush(w_hier, hierarchy_buf, HIERARCHY_SCHEMA)
        _flush(w_bib, bibtex_buf, BIBTEX_SCHEMA)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable parsed.json files, e.g. {skipped[:5]}")

    return {
        "elements_count": counts["elements"],
        "hierarchy_count": counts["hierarchy"],
        "bibtex_count": counts["bibtex"],
        "elements_path": str(elements_path),
        "hierarchy_path": str(hierarchy_path),
        "bibtex_path": str(bibtex_path),
    }