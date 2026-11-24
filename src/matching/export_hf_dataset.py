"""Build the Hugging Face dataset folder: sharded Parquet tables + dataset card.

Input : data/processed/<student_id>/{papers,references,elements,hierarchy,bibtex,predictions}.parquet
Output: data/processed/hf_dataset/
            README.md                       (dataset card with YAML configs)
            data/<table>/part-00000-of-0000N.parquet

Usage:
    python -m src.matching.export_hf_dataset --student-id 24127590
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.matching.select_publications import DEFAULT_OUTPUT_DIR, DEFAULT_STUDENT_ID
from src.matching.train_ranker import DEFAULT_PROCESSED_DIR, METRICS_FILE

TABLES = ["papers", "references", "elements", "hierarchy", "bibtex", "predictions"]

EXPECTED_COLUMNS: Dict[str, List[str]] = {
    "papers": ["arxiv_id", "title", "authors", "submission_date", "revised_dates", "venue"],
    "references": ["arxiv_id", "ref_arxiv_id", "ref_title", "ref_authors", "ref_submission_date", "ref_s2_id"],
    "elements": ["element_id", "arxiv_id", "kind", "content"],
    "hierarchy": ["arxiv_id", "version", "element_id", "parent_id"],
    "bibtex": ["arxiv_id", "citation_key", "entry_type", "title", "authors", "year", "doi", "journal", "url"],
    "predictions": ["arxiv_id", "partition", "citation_key", "groundtruth_arxiv_id"] + [f"cand_{k}" for k in range(1, 6)],
}


def _string_for_null(schema: pa.Schema) -> pa.Schema:
    """Replace columns of Arrow type `null` (all values missing) by string columns."""
    return pa.schema(
        [pa.field(f.name, pa.string() if pa.types.is_null(f.type) else f.type, nullable=True) for f in schema]
    )


def shard_table(src: Path, dest_dir: Path, rows_per_shard: int = 500_000, batch_rows: int = 50_000) -> Tuple[int, int]:
    """Stream a Parquet file into shards part-XXXXX-of-NNNNN.parquet.

    Returns:
        (total rows, number of shards).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("*.parquet"):
        old.unlink()

    parquet_file = pq.ParquetFile(src)
    schema = _string_for_null(parquet_file.schema_arrow)
    paths: List[Path] = []

    def _open() -> pq.ParquetWriter:
        path = dest_dir / f"part-{len(paths):05d}.parquet"
        paths.append(path)
        return pq.ParquetWriter(path, schema, compression="zstd")

    writer = None
    in_shard = total = 0
    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        table = pa.Table.from_batches([batch])
        if table.schema != schema:
            table = table.cast(schema)
        if writer is None or in_shard >= rows_per_shard:
            if writer is not None:
                writer.close()
            writer = _open()
            in_shard = 0
        writer.write_table(table)
        in_shard += len(table)
        total += len(table)
    if writer is None:  # empty table: still publish the schema
        writer = _open()
    writer.close()

    for idx, path in enumerate(paths):
        path.rename(dest_dir / f"part-{idx:05d}-of-{len(paths):05d}.parquet")
    return total, len(paths)


def build_card(counts: Dict[str, int], metrics: Dict) -> str:
    """Dataset card: YAML metadata (configs) + human-readable description."""
    configs = "\n".join(
        f"- config_name: {t}\n  data_files:\n  - split: train\n    path: data/{t}/*.parquet" for t in TABLES
    )
    rows = "\n".join(f"| `{t}` | {counts[t]:,} |" for t in TABLES)

    results = ""
    if metrics:
        lines = ["| Partition | Origin | Labelled entries | MRR@5 | MRR@5 (positives only) | Hit@1 | Hit@5 |", "|---|---|---|---|---|---|---|"]
        for part in ("valid", "test"):
            for origin in ("manual", "auto", "all"):
                s = metrics["model"][part][origin]
                lines.append(
                    f"| {part} | {origin} | {s['n_labeled']} | {s['mrr5']:.4f} | "
                    f"{s['mrr5_positive_only']:.4f} | {s['hit@1']:.3f} | {s['hit@5']:.3f} |"
                )
        base = metrics["baseline_heuristic"]["test"]["all"]
        results = (
            "\n## Matching results\n\nLightGBM LambdaRank over (entry, reference) pairs of the same publication.\n\n"
            + "\n".join(lines)
            + f"\n\nRule-based baseline on the test partition: MRR@5 = {base['mrr5']:.4f}.\n"
            "Automatic labels come from the same rule-based matcher that provides some features, "
            "so the `auto` rows are optimistic; the `manual` rows are the honest estimate.\n"
        )

    return f"""---
pretty_name: arXiv LaTeX parsing and reference matching (Introduction to Data Science, Group 13)
language:
- en
license: other
task_categories:
- text-retrieval
tags:
- arxiv
- latex
- entity-resolution
- reference-matching
configs:
{configs}
---

# arXiv LaTeX parsing and reference matching

Dataset built for the Introduction to Data Science course project: crawled arXiv papers
(all versions), parsed LaTeX hierarchy, unified BibTeX entries, and reference-matching
predictions. Paper contents keep the licences chosen by their authors on arXiv.

## Tables

| Config | Rows |
|---|---|
{rows}

* `papers`: crawled paper metadata, one row per paper.
* `references`: Semantic Scholar references that carry an arXiv ID, one row per (citing paper, reference).
* `elements`: parsed document elements, deduplicated across versions (`kind`: document, heading, sentence, formula, figure, itemize, item).
* `hierarchy`: node-parent edges of each version.
* `bibtex`: unified BibTeX entries after deduplication.
* `predictions`: one row per BibTeX entry of each publication used for matching.
  `partition` is `train`, `valid` or `test`; `groundtruth_arxiv_id` is the true reference
  when known (null when unlabelled or when no crawled reference matches);
  `cand_1`..`cand_5` are the model's ranked candidates (best first).

## Labels

Manual labels: at least 5 publications with >= 20 labelled entries each. Automatic labels:
a regex plus string-similarity matcher applied to a random sample of the remaining
publications. Test and validation sets each hold 1 manual and 1 automatic publication.
{results}
## Usage

```python
from datasets import load_dataset
ds = load_dataset("<user>/<repo>", "predictions", split="train", streaming=True)
print(next(iter(ds)))
```
"""


def export_dataset(
    student_id: str,
    processed_dir: Path,
    output_dir: Path,
    metrics_path: Path,
    rows_per_shard: int = 500_000,
) -> Dict[str, int]:
    """Validate the six tables, shard them and write the dataset card."""
    src_dir = processed_dir / student_id
    for table in TABLES:  # fail early, before writing anything
        path = src_dir / f"{table}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing table: {path}")
        names = pq.ParquetFile(path).schema_arrow.names
        if names != EXPECTED_COLUMNS[table]:
            raise ValueError(f"{table}: columns {names} differ from expected {EXPECTED_COLUMNS[table]}")

    counts: Dict[str, int] = {}
    for table in TABLES:
        total, shards = shard_table(src_dir / f"{table}.parquet", output_dir / "data" / table, rows_per_shard)
        counts[table] = total
        print(f"{table:12s} rows={total:10,d} shards={shards}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    (output_dir / "README.md").write_text(build_card(counts, metrics), encoding="utf-8")
    return counts


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the Hugging Face dataset folder")
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR / "hf_dataset")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_OUTPUT_DIR / METRICS_FILE)
    parser.add_argument("--rows-per-shard", type=int, default=500_000)
    args = parser.parse_args()
    export_dataset(args.student_id, args.processed_dir, args.output_dir, args.metrics, args.rows_per_shard)
    print(f"\nDataset folder ready: {args.output_dir}")


if __name__ == "__main__":
    main()