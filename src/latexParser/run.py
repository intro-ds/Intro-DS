"""SCRUM-36: Incremental & parallel Parse Runner that works alongside the crawler.

Usage examples:
  1. One-shot parse of all currently downloaded papers:
       python3 -m src.latexParser.run --student-id 24127101

  2. Parallel mode (runs alongside the crawler until all assigned papers finish):
       python3 -m src.latexParser.run --student-id 24127101 --follow

  3. Measured with the benchmark utility:
       python3 -m src.utils.benchmark --label m2_parser \
           --output-dir data/processed -- python3 -m src.latexParser.run --student-id 24127101
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.crawling.source_and_references import (
    CONFIG_PATH,
    DISCOVERY_FILE,
    DISCOVERY_MANIFEST,
    RAW_DIR,
    ROOT,
    SOURCE_DIR,
    SOURCE_MANIFEST,
    assigned_base_ids,
)
from src.latexParser.file_resolver import LatexFileResolver
from src.latexParser.structure_extractor import PublicationHierarchyBuilder

PROCESSED_DIR = ROOT / "data" / "processed"
ELEMENTS_JSONL = PROCESSED_DIR / "elements.jsonl"
HIERARCHY_JSONL = PROCESSED_DIR / "hierarchy.jsonl"
BIBTEX_JSONL = PROCESSED_DIR / "bibtex.jsonl"
ELEMENTS_PARQUET = PROCESSED_DIR / "elements.parquet"
HIERARCHY_PARQUET = PROCESSED_DIR / "hierarchy.parquet"
BIBTEX_PARQUET = PROCESSED_DIR / "bibtex.parquet"
PARSE_MANIFEST = PROCESSED_DIR / "parse_manifest.csv"

PARSE_MANIFEST_FIELDS = [
    "arxiv_id",
    "version",
    "status",
    "bibtex_raw_count",
    "bibtex_dedup_count",
    "elements_before_dedup",
    "elements_after_dedup",
    "hierarchy_edges",
    "included_tex_files",
    "ignored_tex_files",
    "elapsed_seconds",
    "error",
]

ELEMENTS_SCHEMA = pa.schema([
    ("element_id", pa.string()),
    ("arxiv_id", pa.string()),
    ("kind", pa.dictionary(pa.int8(), pa.string())),
    ("content", pa.string()),
])

HIERARCHY_SCHEMA = pa.schema([
    ("arxiv_id", pa.string()),
    ("version", pa.int32()),
    ("element_id", pa.string()),
    ("parent_id", pa.string()),
])

BIBTEX_SCHEMA = pa.schema([
    ("arxiv_id", pa.string()),
    ("citation_key", pa.string()),
    ("entry_type", pa.dictionary(pa.int8(), pa.string())),
    ("title", pa.string()),
    ("authors", pa.list_(pa.string())),
    ("year", pa.int32()),
    ("doi", pa.string()),
    ("journal", pa.string()),
    ("url", pa.string()),
])

TERMINAL_SOURCE_STATUSES = {"success", "no_source", "no_tex_or_bib", "non_archive"}


def extract_bibtex_stub(
    compiled_tex: str, version_dir: Path, arxiv_id: str = "", version: int = 1
) -> List[dict]:
    return []


def dedup_bibtex_stub(
    raw_entries: List[dict], arxiv_id: str = ""
) -> Tuple[List[dict], Dict[str, str]]:
    return [], {}


def rewrite_cites_stub(compiled_tex: str, key_map: Dict[str, str]) -> str:
    return compiled_tex


def _resolve_bibtex_hooks() -> Tuple[Callable, Callable, Callable]:
    extract_fn: Callable = extract_bibtex_stub
    dedup_fn: Callable = dedup_bibtex_stub
    rewrite_fn: Callable = rewrite_cites_stub

    for mod_name in ("src.latexParser.bibtex_extractor", "src.latexParser.bibtex"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "extract_bibtex"):
                extract_fn = getattr(mod, "extract_bibtex")
                break
        except ModuleNotFoundError:
            continue

    for mod_name in (
        "src.latexParser.bibtex_dedup",
        "src.latexParser.deduplicator",
        "src.latexParser.bibtex_extractor",
        "src.latexParser.bibtex",
    ):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "dedup_bibtex"):
                dedup_fn = getattr(mod, "dedup_bibtex")
            if hasattr(mod, "rewrite_cites"):
                rewrite_fn = getattr(mod, "rewrite_cites")
        except ModuleNotFoundError:
            continue

    return extract_fn, dedup_fn, rewrite_fn


def _normalize_extract_result(result) -> Tuple[List[dict], int]:
    if isinstance(result, tuple) and len(result) == 2:
        entries, raw_count = result
        entries_list = list(entries or [])
        return entries_list, int(raw_count)
    entries_list = list(result or [])
    return entries_list, len(entries_list)


def _format_bibtex_rows(entries: Iterable[dict], arxiv_id: str) -> List[dict]:
    formatted: List[dict] = []
    for entry in entries:
        key = str(entry.get("citation_key") or "").strip()
        if not key:
            continue
        year_val = entry.get("year")
        try:
            year_int = int(year_val) if year_val not in (None, "") else None
        except (TypeError, ValueError):
            year_int = None
        authors = entry.get("authors") or []
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(" and ") if a.strip()]
        formatted.append({
            "arxiv_id": str(entry.get("arxiv_id") or arxiv_id),
            "citation_key": key,
            "entry_type": str(entry.get("entry_type") or "misc"),
            "title": entry.get("title"),
            "authors": [str(a) for a in authors],
            "year": year_int,
            "doi": entry.get("doi"),
            "journal": entry.get("journal"),
            "url": entry.get("url"),
        })
    return formatted


def _normalize_dedup_result(
    result, arxiv_id: str
) -> Tuple[List[dict], Dict[str, str]]:
    if isinstance(result, tuple) and len(result) == 2:
        entries, key_map = result
        return _format_bibtex_rows(entries or [], arxiv_id), dict(key_map or {})
    return _format_bibtex_rows(result or [], arxiv_id), {}


def _filter_base_ids(
    student_id: Optional[str] = None,
    prefix: Optional[str] = None,
    limit: Optional[int] = None,
    config_path: Path = CONFIG_PATH,
    source_manifest: Path = SOURCE_MANIFEST,
    follow: bool = False,
) -> List[str]:
    base_ids = assigned_base_ids(config_path)
    if student_id:
        with Path(config_path).open("r", encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("student_id") == student_id]
        if not rows:
            raise ValueError(f"student_id {student_id!r} not found in {config_path}")
        base_ids = [
            f"{r['start_month'].replace('-', '')[2:]}.{num:05d}"
            for r in rows
            for num in range(int(r["start_id"]), int(r["end_id"]) + 1)
        ]
    elif not prefix and not follow and Path(source_manifest).exists():
        manifest_ids: Set[str] = set()
        with Path(source_manifest).open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                bid = r.get("arxiv_id", "")
                if bid:
                    manifest_ids.add(bid)
        if manifest_ids:
            base_ids = [bid for bid in base_ids if bid in manifest_ids]

    if prefix:
        base_ids = [bid for bid in base_ids if bid.startswith(f"{prefix}.")]
    if limit is not None:
        base_ids = base_ids[:limit]
    return base_ids


def _load_discovered_histories(wanted: Set[str]) -> Dict[str, List[int]]:
    histories: Dict[str, List[int]] = {}
    paths = [
        p for p in sorted(RAW_DIR.rglob("arxiv_history_*.jsonl"))
        if p != DISCOVERY_FILE
    ]
    if DISCOVERY_FILE.exists():
        paths.append(DISCOVERY_FILE)

    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                base_id = record.get("base_id")
                versions = record.get("versions") or []
                if base_id in wanted and versions:
                    v_nums = []
                    for v_str in versions:
                        if "v" in v_str:
                            try:
                                v_nums.append(int(v_str.rsplit("v", 1)[-1]))
                            except ValueError:
                                pass
                    if v_nums:
                        histories[base_id] = sorted(set(v_nums))
    return histories


def _load_discovery_not_found(wanted: Set[str]) -> Set[str]:
    not_found: Set[str] = set()
    if not DISCOVERY_MANIFEST.exists():
        return not_found
    with DISCOVERY_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            bid = row.get("base_id", "")
            if bid in wanted and row.get("status") == "not_found":
                not_found.add(bid)
            elif bid in not_found and row.get("status") == "success":
                not_found.discard(bid)
    return not_found


def _load_downloaded_versions(
    source_manifest: Path, source_dir: Path, histories: Dict[str, List[int]]
) -> Dict[Tuple[str, int], str]:
    ready: Dict[Tuple[str, int], str] = {}
    if source_manifest.exists():
        with source_manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                bid = row.get("arxiv_id", "")
                status = row.get("status", "")
                try:
                    ver = int(row.get("version", "0"))
                except ValueError:
                    continue
                if status in TERMINAL_SOURCE_STATUSES:
                    ready[(bid, ver)] = status

    for bid, v_list in histories.items():
        pub_dir = source_dir / bid
        if not pub_dir.exists():
            continue
        for ver in v_list:
            if (bid, ver) in ready:
                continue
            ver_dir = pub_dir / f"v{ver}"
            if ver_dir.is_dir():
                has_tex = any(ver_dir.rglob("*.tex")) or any(ver_dir.rglob("*.bib"))
                ready[(bid, ver)] = "success" if has_tex else "no_tex_or_bib"
    return ready


def _load_completed_publications(
    manifest_path: Path, histories: Dict[str, List[int]]
) -> Set[str]:
    if not manifest_path.exists():
        return set()
    version_statuses: Dict[str, Dict[int, str]] = defaultdict(dict)
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            bid = row.get("arxiv_id", "")
            status = row.get("status", "")
            ver_raw = row.get("version", "")
            if not bid:
                continue
            if ver_raw == "":
                if status in {"parsed", "empty", "success"}:
                    version_statuses[bid][-1] = status
            else:
                try:
                    version_statuses[bid][int(ver_raw)] = status
                except ValueError:
                    continue

    completed: Set[str] = set()
    for bid, v_map in version_statuses.items():
        if -1 in v_map:
            completed.add(bid)
            continue
        expected_versions = histories.get(bid)
        if expected_versions and all(
            v_map.get(v) in {"parsed", "empty"} for v in expected_versions
        ):
            completed.add(bid)
    return completed


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _append_manifest_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PARSE_MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _call_extract_bibtex(
    extract_fn: Callable, compiled_tex: str, ver_dir: Path, base_id: str, ver: int
):
    try:
        return extract_fn(compiled_tex, ver_dir, arxiv_id=base_id, version=ver)
    except TypeError:
        try:
            return extract_fn(compiled_tex, ver_dir)
        except TypeError:
            return extract_fn(compiled_tex)


def _call_dedup_bibtex(dedup_fn: Callable, raw_entries: List[dict], base_id: str):
    try:
        return dedup_fn(raw_entries, arxiv_id=base_id)
    except TypeError:
        return dedup_fn(raw_entries)


def parse_single_publication(
    base_id: str,
    versions: List[int],
    source_dir: Path,
    resolver: LatexFileResolver,
    extract_fn: Optional[Callable] = None,
    dedup_fn: Optional[Callable] = None,
    rewrite_fn: Optional[Callable] = None,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    if extract_fn is None or dedup_fn is None or rewrite_fn is None:
        default_extract, default_dedup, default_rewrite = _resolve_bibtex_hooks()
        extract_fn = extract_fn or default_extract
        dedup_fn = dedup_fn or default_dedup
        rewrite_fn = rewrite_fn or default_rewrite

    builder = PublicationHierarchyBuilder(base_id)
    raw_bibtex_by_version: Dict[int, List[dict]] = {}
    pass1_records: List[dict] = []

    for ver in sorted(versions):
        ver_started = time.perf_counter()
        ver_dir = source_dir / base_id / f"v{ver}"
        try:
            res = resolver.resolve_version(ver_dir)
            compiled_tex = str(res.get("compiled_tex") or "")
            is_empty = bool(res.get("is_empty")) or not compiled_tex.strip()
            ver_raw_entries: List[dict] = []
            ver_raw_count = 0
            if not is_empty:
                raw_res = _call_extract_bibtex(
                    extract_fn, compiled_tex, ver_dir, base_id, ver
                )
                ver_raw_entries, ver_raw_count = _normalize_extract_result(raw_res)
                raw_bibtex_by_version[ver] = ver_raw_entries

            pass1_records.append({
                "version": ver,
                "is_empty": is_empty,
                "compiled_tex": compiled_tex,
                "included_count": len(res.get("included_files") or []),
                "ignored_count": len(res.get("ignored_files") or []),
                "bibtex_raw_count": ver_raw_count,
                "elapsed": time.perf_counter() - ver_started,
                "error": "",
            })
        except Exception as exc:
            pass1_records.append({
                "version": ver,
                "is_empty": True,
                "compiled_tex": "",
                "included_count": 0,
                "ignored_count": 0,
                "bibtex_raw_count": 0,
                "elapsed": time.perf_counter() - ver_started,
                "error": str(exc),
            })

    bibtex_rows: List[dict] = []
    key_map: Dict[str, str] = {}
    dedup_error = ""
    try:
        dedup_res = _call_dedup_bibtex(dedup_fn, raw_bibtex_by_version, base_id)
        bibtex_rows, key_map = _normalize_dedup_result(dedup_res, base_id)
    except Exception as exc:
        dedup_error = str(exc)

    manifest_rows: List[dict] = []
    for rec in pass1_records:
        ver = rec["version"]
        p2_started = time.perf_counter()
        if rec["error"] or dedup_error:
            err_msg = rec["error"] or dedup_error
            manifest_rows.append({
                "arxiv_id": base_id,
                "version": ver,
                "status": "failed",
                "bibtex_raw_count": rec["bibtex_raw_count"],
                "bibtex_dedup_count": len(bibtex_rows),
                "elements_before_dedup": 0,
                "elements_after_dedup": len(builder.elements_rows),
                "hierarchy_edges": len(builder.hierarchy_rows),
                "included_tex_files": rec["included_count"],
                "ignored_tex_files": rec["ignored_count"],
                "elapsed_seconds": round(
                    rec["elapsed"] + (time.perf_counter() - p2_started), 4
                ),
                "error": err_msg,
            })
            continue

        try:
            elements_before_total = len(builder.elements_rows)
            edges_before = len(builder.hierarchy_rows)
            if rec["is_empty"]:
                builder.parse_version(ver, "")
                ver_status = "empty"
                ver_elements_before = 0
            else:
                rewritten_tex = rewrite_fn(rec["compiled_tex"], key_map)
                builder.parse_version(ver, rewritten_tex)
                ver_status = "parsed"
                ver_elements_before = 1 + (len(builder.hierarchy_rows) - edges_before)

            ver_edges = len(builder.hierarchy_rows) - edges_before
            ver_new_elements = len(builder.elements_rows) - elements_before_total
            manifest_rows.append({
                "arxiv_id": base_id,
                "version": ver,
                "status": ver_status,
                "bibtex_raw_count": rec["bibtex_raw_count"],
                "bibtex_dedup_count": len(bibtex_rows),
                "elements_before_dedup": ver_elements_before,
                "elements_after_dedup": ver_new_elements,
                "hierarchy_edges": ver_edges,
                "included_tex_files": rec["included_count"],
                "ignored_tex_files": rec["ignored_count"],
                "elapsed_seconds": round(
                    rec["elapsed"] + (time.perf_counter() - p2_started), 4
                ),
                "error": "",
            })
        except Exception as exc:
            manifest_rows.append({
                "arxiv_id": base_id,
                "version": ver,
                "status": "failed",
                "bibtex_raw_count": rec["bibtex_raw_count"],
                "bibtex_dedup_count": len(bibtex_rows),
                "elements_before_dedup": 0,
                "elements_after_dedup": 0,
                "hierarchy_edges": 0,
                "included_tex_files": rec["included_count"],
                "ignored_tex_files": rec["ignored_count"],
                "elapsed_seconds": round(
                    rec["elapsed"] + (time.perf_counter() - p2_started), 4
                ),
                "error": str(exc),
            })

    return builder.elements_rows, builder.hierarchy_rows, bibtex_rows, manifest_rows


def export_parquet_tables(
    elements_jsonl: Path = ELEMENTS_JSONL,
    hierarchy_jsonl: Path = HIERARCHY_JSONL,
    bibtex_jsonl: Path = BIBTEX_JSONL,
    elements_parquet: Path = ELEMENTS_PARQUET,
    hierarchy_parquet: Path = HIERARCHY_PARQUET,
    bibtex_parquet: Path = BIBTEX_PARQUET,
) -> Tuple[int, int, int]:
    elements_parquet.parent.mkdir(parents=True, exist_ok=True)

    el_by_id: Dict[str, dict] = {}
    if elements_jsonl.exists():
        with elements_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    el_by_id[row["element_id"]] = row

    hi_seen: Set[Tuple[str, int, str, str]] = set()
    hi_rows: List[dict] = []
    if hierarchy_jsonl.exists():
        with hierarchy_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = (
                        row["arxiv_id"],
                        int(row["version"]),
                        row["element_id"],
                        row["parent_id"],
                    )
                    if key not in hi_seen:
                        hi_seen.add(key)
                        hi_rows.append(row)

    bib_by_key: Dict[Tuple[str, str], dict] = {}
    if bibtex_jsonl.exists():
        with bibtex_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    bib_by_key[(row["arxiv_id"], row["citation_key"])] = row

    el_list = list(el_by_id.values())
    bib_list = list(bib_by_key.values())
    el_table = pa.Table.from_pylist(el_list, schema=ELEMENTS_SCHEMA)
    hi_table = pa.Table.from_pylist(hi_rows, schema=HIERARCHY_SCHEMA)
    bib_table = pa.Table.from_pylist(bib_list, schema=BIBTEX_SCHEMA)

    for table, target in [
        (el_table, elements_parquet),
        (hi_table, hierarchy_parquet),
        (bib_table, bibtex_parquet),
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.stem}-", suffix=".parquet", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, target)

    return len(el_list), len(hi_rows), len(bib_list)


def run_parser(
    base_ids: List[str],
    source_dir: Path = SOURCE_DIR,
    source_manifest: Path = SOURCE_MANIFEST,
    output_dir: Path = PROCESSED_DIR,
    follow: bool = False,
    poll_interval: float = 5.0,
    idle_timeout: float = 900.0,
    write_parquet: bool = True,
) -> int:
    output_dir = Path(output_dir)
    elements_jsonl = output_dir / "elements.jsonl"
    hierarchy_jsonl = output_dir / "hierarchy.jsonl"
    bibtex_jsonl = output_dir / "bibtex.jsonl"
    elements_parquet = output_dir / "elements.parquet"
    hierarchy_parquet = output_dir / "hierarchy.parquet"
    bibtex_parquet = output_dir / "bibtex.parquet"
    parse_manifest = output_dir / "parse_manifest.csv"

    wanted = set(base_ids)
    resolver = LatexFileResolver()
    extract_fn, dedup_fn, rewrite_fn = _resolve_bibtex_hooks()
    total_newly_parsed = 0
    last_activity = time.monotonic()

    while True:
        histories = _load_discovered_histories(wanted)
        not_found_ids = _load_discovery_not_found(wanted)
        ready_versions = _load_downloaded_versions(
            source_manifest, source_dir, histories
        )
        completed_ids = _load_completed_publications(parse_manifest, histories)

        batch_parsed = 0
        for base_id in base_ids:
            if base_id in completed_ids or base_id in not_found_ids:
                continue
            versions = histories.get(base_id)
            if not versions:
                continue
            if not all((base_id, v) in ready_versions for v in versions):
                continue

            el_rows, hi_rows, bib_rows, m_rows = parse_single_publication(
                base_id,
                versions,
                source_dir,
                resolver,
                extract_fn=extract_fn,
                dedup_fn=dedup_fn,
                rewrite_fn=rewrite_fn,
            )
            _append_jsonl(elements_jsonl, el_rows)
            _append_jsonl(hierarchy_jsonl, hi_rows)
            _append_jsonl(bibtex_jsonl, bib_rows)
            _append_manifest_rows(parse_manifest, m_rows)

            if all(r["status"] in {"parsed", "empty"} for r in m_rows):
                completed_ids.add(base_id)
            batch_parsed += 1
            total_newly_parsed += 1

            statuses = ",".join(f"v{r['version']}:{r['status']}" for r in m_rows)
            print(
                f"Parsed [{len(completed_ids)}/{len(histories)}]: {base_id} "
                f"[{statuses}] ({len(bib_rows)} bibtex, {len(el_rows)} elements)"
            )

        if batch_parsed > 0:
            last_activity = time.monotonic()

        if not follow:
            break

        if len(completed_ids) + len(not_found_ids) >= len(wanted):
            print("All assigned publications have been processed!")
            break

        if time.monotonic() - last_activity > idle_timeout:
            print(f"No new ready papers for {idle_timeout}s; stopping follow loop.")
            break

        time.sleep(poll_interval)

    if write_parquet:
        n_el, n_hi, n_bib = export_parquet_tables(
            elements_jsonl=elements_jsonl,
            hierarchy_jsonl=hierarchy_jsonl,
            bibtex_jsonl=bibtex_jsonl,
            elements_parquet=elements_parquet,
            hierarchy_parquet=hierarchy_parquet,
            bibtex_parquet=bibtex_parquet,
        )
        print(
            f"Exported Parquet tables -> elements: {n_el} rows, "
            f"hierarchy: {n_hi} rows, bibtex: {n_bib} rows."
        )
    return total_newly_parsed


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the Milestone 2 LaTeX hierarchy & BibTeX parser."
    )
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    ap.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    ap.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--student-id", type=str)
    ap.add_argument("--prefix", type=str)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--poll", "--follow", dest="follow", action="store_true")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--idle-timeout", type=float, default=900.0)
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    base_ids = _filter_base_ids(
        student_id=args.student_id,
        prefix=args.prefix,
        limit=args.limit,
        config_path=args.config,
        source_manifest=args.source_manifest,
        follow=args.follow,
    )
    run_parser(
        base_ids=base_ids,
        source_dir=args.source_dir,
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        follow=args.follow,
        poll_interval=args.poll_interval,
        idle_timeout=args.idle_timeout,
        write_parquet=not args.no_parquet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())