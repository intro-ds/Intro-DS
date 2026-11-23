"""Candidate files for manual labelling and collection of the manual labels (Milestone 3).

Two sub-commands:

  generate  For each selected publication, write one CSV with one row per BibTeX entry and
            the top-5 suggested references (ranked by the string-similarity score of
            auto_labeler.py). The annotator fills the `label` column.
  collect   Read the filled CSVs, validate them, write labels_manual.csv and check the
            mandatory quantities (>= 5 publications, >= 20 labelled entries each,
            >= 100 labelled entries in total, per member).

Label format (column `label` of data/interim/matching/manual_labels/<arxiv_id>.csv):
    1..5          the entry corresponds to suggestion cand<k> of that row
    <arXiv ID>    the entry corresponds to that reference (look it up in
                  manual_labels/references/<arxiv_id>.csv when it is not in the top 5)
    NONE          no crawled reference matches this entry (counts as a labelled entry)
    (blank)       not labelled yet (ignored)
Tip: use rank numbers 1..5 when possible; spreadsheets may corrupt arXiv IDs such as
"2001.10000" into numbers.

Usage:
    python -m src.matching.build_candidates generate
    python -m src.matching.build_candidates collect --members 1
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.matching.auto_labeler import rank_candidates
from src.matching.select_publications import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_INTERMEDIATE_DIR,
    DEFAULT_OUTPUT_DIR,
    MANUAL_PUBLICATIONS_FILE,
    load_entries,
    load_references,
    read_id_file,
)

LABELS_DIR_NAME = "manual_labels"
MANUAL_LABELS_FILE = "labels_manual.csv"
LABEL_COLUMN = "label"
NONE_TOKENS = {"none", "null", "no", "-"}
CSV_ENCODING = "utf-8-sig"  # BOM so that spreadsheet tools display accents correctly


def _candidate_columns(top_k: int) -> List[str]:
    cols: List[str] = []
    for k in range(1, top_k + 1):
        cols += [f"cand{k}_id", f"cand{k}_score", f"cand{k}_title", f"cand{k}_authors"]
    return cols


def _entry_columns() -> List[str]:
    return [
        "arxiv_id",
        "citation_key",
        "entry_title",
        "entry_authors",
        "entry_year",
        "entry_journal",
        "entry_doi",
        "entry_url",
    ]


def generate_candidates(
    arxiv_id: str,
    intermediate_dir: Path = DEFAULT_INTERMEDIATE_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_entries: int = 30,
    top_k: int = 5,
    seed: int = 42,
    force: bool = False,
) -> Dict[str, Any]:
    """Write the labelling CSV of one publication.

    Entries are sampled at random (seeded) among those with a title or authors, so that
    the manual labels contain both positives and "no match" rows. max_entries=0 keeps all.
    An existing file is never overwritten unless force=True (protects manual work).
    """
    labels_dir = output_dir / LABELS_DIR_NAME
    path = labels_dir / f"{arxiv_id}.csv"
    if path.exists() and not force:
        return {"arxiv_id": arxiv_id, "status": "skipped_exists", "path": str(path)}

    entries = [e for e in load_entries(arxiv_id, intermediate_dir) if e.get("title") or e.get("authors")]
    refs = load_references(arxiv_id, corpus_dir)
    if not entries or not refs:
        return {"arxiv_id": arxiv_id, "status": "skipped_no_data", "n_entries": len(entries), "n_refs": len(refs)}

    if max_entries and len(entries) > max_entries:
        rng = random.Random(f"{seed}-{arxiv_id}")
        keep = sorted(rng.sample(range(len(entries)), max_entries))
        entries = [entries[i] for i in keep]

    labels_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _entry_columns() + _candidate_columns(top_k) + [LABEL_COLUMN]
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row: Dict[str, Any] = {
                "arxiv_id": arxiv_id,
                "citation_key": entry.get("citation_key", ""),
                "entry_title": entry.get("title") or "",
                "entry_authors": "; ".join(entry.get("authors") or []),
                "entry_year": entry.get("year") or "",
                "entry_journal": entry.get("journal") or "",
                "entry_doi": entry.get("doi") or "",
                "entry_url": entry.get("url") or "",
                LABEL_COLUMN: "",
            }
            for k, cand in enumerate(rank_candidates(entry, refs, top_k), start=1):
                row[f"cand{k}_id"] = cand["ref_arxiv_id"]
                row[f"cand{k}_score"] = cand["score"]
                row[f"cand{k}_title"] = cand["ref_title"]
                row[f"cand{k}_authors"] = "; ".join(cand["ref_authors"][:3])
            writer.writerow(row)

    # Full reference list for look-ups when the right reference is not in the top-k
    ref_dir = labels_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    with (ref_dir / f"{arxiv_id}.csv").open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_arxiv_id", "ref_title", "ref_authors", "ref_submission_date"])
        for r in refs:
            writer.writerow(
                [r["ref_arxiv_id"], r["ref_title"], "; ".join(r["ref_authors"]), r["ref_submission_date"]]
            )

    return {"arxiv_id": arxiv_id, "status": "written", "n_rows": len(entries), "n_refs": len(refs), "path": str(path)}


def _normalize_ref_id(raw: str) -> str:
    """Strip an 'arXiv:' prefix and a version suffix from a typed arXiv ID."""
    raw = re.sub(r"^arxiv:\s*", "", raw.strip(), flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", raw)


def collect_manual_labels(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    members: int = 1,
    min_publications: int = 5,
    min_entries_per_publication: int = 20,
    min_total_entries: int = 100,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Validate the filled CSVs, write labels_manual.csv and check the quantities.

    A publication qualifies as manually labelled when it has >= min_entries_per_publication
    labelled entries. The requirement is met when qualifying publications >=
    min_publications * members and their labelled entries >= min_total_entries * members.
    """
    labels_dir = output_dir / LABELS_DIR_NAME
    rows: List[Dict[str, str]] = []
    errors: List[str] = []
    per_publication: Dict[str, int] = {}

    for path in sorted(labels_dir.glob("*.csv")):
        aid = path.stem
        ref_ids = {r["ref_arxiv_id"] for r in load_references(aid, corpus_dir)}
        count = 0
        with path.open("r", encoding=CSV_ENCODING, newline="") as f:
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                raw = (row.get(LABEL_COLUMN) or "").strip()
                if not raw:
                    continue  # not labelled yet
                key = row.get("citation_key", "")
                if raw.lower() in NONE_TOKENS:
                    truth: Optional[str] = ""
                elif raw.isdigit() and 1 <= int(raw) <= top_k:
                    truth = (row.get(f"cand{raw}_id") or "").strip() or None
                else:
                    candidate = _normalize_ref_id(raw)
                    truth = candidate if candidate in ref_ids else None
                if truth is None:
                    errors.append(f"{path.name}:{line_no} invalid label '{raw}' for '{key}'")
                    continue
                rows.append(
                    {"arxiv_id": aid, "citation_key": key, "groundtruth_arxiv_id": truth, "source": "manual"}
                )
                count += 1
        per_publication[aid] = count

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / MANUAL_LABELS_FILE).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arxiv_id", "citation_key", "groundtruth_arxiv_id", "source"])
        writer.writeheader()
        writer.writerows(rows)

    qualifying = {a: n for a, n in per_publication.items() if n >= min_entries_per_publication}
    total_qualifying = sum(qualifying.values())
    return {
        "per_publication": per_publication,
        "qualifying_publications": len(qualifying),
        "required_publications": min_publications * members,
        "qualifying_entries": total_qualifying,
        "required_entries": min_total_entries * members,
        "entries_positive": sum(1 for r in rows if r["groundtruth_arxiv_id"]),
        "entries_none": sum(1 for r in rows if not r["groundtruth_arxiv_id"]),
        "errors": errors,
        "requirement_met": len(qualifying) >= min_publications * members
        and total_qualifying >= min_total_entries * members,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manual-labelling candidate files")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Write the CSV files to label")
    gen.add_argument("--ids", nargs="*", help="arXiv IDs (default: manual_publications.txt)")
    gen.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    gen.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    gen.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    gen.add_argument("--max-entries", type=int, default=30, help="Rows per publication (0 = all)")
    gen.add_argument("--top-k", type=int, default=5)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--force", action="store_true", help="Overwrite existing label files")

    col = sub.add_parser("collect", help="Validate filled CSVs and write labels_manual.csv")
    col.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    col.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    col.add_argument("--members", type=int, default=1, help="Group size k the quantities are multiplied by")

    args = parser.parse_args()

    if args.command == "generate":
        ids = args.ids or read_id_file(args.output_dir / MANUAL_PUBLICATIONS_FILE)
        if not ids:
            print("No publications given. Run select_publications first or pass --ids.")
            return
        for aid in ids:
            res = generate_candidates(
                aid, args.intermediate_dir, args.corpus_dir, args.output_dir,
                args.max_entries, args.top_k, args.seed, args.force,
            )
            print(res)
    else:
        res = collect_manual_labels(args.output_dir, args.corpus_dir, args.members)
        for aid, n in res["per_publication"].items():
            print(f"  {aid}: {n} labelled entries")
        print(f"qualifying publications: {res['qualifying_publications']} / {res['required_publications']}")
        print(f"qualifying entries:      {res['qualifying_entries']} / {res['required_entries']}")
        print(f"positives / none rows:   {res['entries_positive']} / {res['entries_none']}")
        for err in res["errors"]:
            print("  ERROR:", err)
        print("REQUIREMENT:", "PASS" if res["requirement_met"] else "NOT MET YET")


if __name__ == "__main__":
    main()