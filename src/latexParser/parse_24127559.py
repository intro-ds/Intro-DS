"""LaTeX parsing pipeline runner for Nguyen Thanh Toan (student ID: 24127559).

Assigned range: 2025-12, IDs 3231 to 8230 (2512.03231 to 2512.08230).

Usage:
    python -m src.latexParser.parse_24127559 [--stage STAGE] [--offset OFFSET] [--limit LIMIT] [--force]

Stages:
    all     Execute full parsing and export stages (default)
    parse   Parse LaTeX sources and save intermediate structured JSONs
    export  Export Parquet tables (elements, hierarchy, bibtex) from existing intermediate JSONs
"""

import argparse
from pathlib import Path
import time
from typing import List

import psutil

from src.crawling.config_reader import load_member_range
from src.latexParser.intermediate_writer import DEFAULT_INTERMEDIATE_DIR
from src.latexParser.parquet_exporter import export_parquet_tables
from src.latexParser.parse_pipeline import parse_corpus

STUDENT_ID = "24127559"
CORPUS_DIR = Path("data/raw/corpus")
INTERMEDIATE_DIR = DEFAULT_INTERMEDIATE_DIR
PROCESSED_DIR = Path("data/processed")


def _get_ram_usage_mb() -> float:
    """Get current resident set size (RAM) in megabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_pipeline(
    stage: str = "all",
    limit: int = 0,
    offset: int = 0,
    force: bool = False,
) -> None:
    """Execute parsing stages for student 24127559.

    Args:
        stage: Pipeline stage to execute ('all', 'parse', 'export').
        limit: If > 0, restrict execution to N assigned papers.
        offset: Starting index within assigned IDs (default: 0).
        force: If True, re-parse papers even if already cached.
    """
    print(f"=== Starting LaTeX Parsing Pipeline for Student {STUDENT_ID} ===")
    t_start = time.perf_counter()
    ram_start = _get_ram_usage_mb()

    # 1. Load assigned range
    member_info = load_member_range(STUDENT_ID)
    all_arxiv_ids: List[str] = member_info["arxiv_ids"]
    total_assigned = len(all_arxiv_ids)

    start_idx = max(0, offset)
    end_idx = start_idx + limit if limit > 0 else total_assigned
    arxiv_ids: List[str] = all_arxiv_ids[start_idx:end_idx]

    if not arxiv_ids:
        print(
            f"No papers selected for offset={offset}, limit={limit} "
            f"(total assigned: {total_assigned})."
        )
        return

    if offset > 0 or limit > 0:
        print(
            f"Applying range: processing {len(arxiv_ids)} papers "
            f"(indices {start_idx} to {start_idx + len(arxiv_ids) - 1} of {total_assigned}): "
            f"{arxiv_ids[0]} -> {arxiv_ids[-1]}."
        )
    else:
        print(f"Total papers assigned: {total_assigned} ({arxiv_ids[0]} -> {arxiv_ids[-1]}).")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Parse LaTeX sources
    if stage in ("all", "parse"):
        print("\n--- [Stage 1/2] Parsing Hierarchical LaTeX Structures ---")
        t0 = time.perf_counter()
        parse_res = parse_corpus(
            arxiv_ids,
            corpus_dir=CORPUS_DIR,
            intermediate_dir=INTERMEDIATE_DIR,
            force=force,
        )
        print(
            f"Parsing completed in {time.perf_counter() - t0:.2f}s | "
            f"Completed: {parse_res['completed_papers']}, "
            f"Cached: {parse_res['cached_papers']}, "
            f"Failed: {parse_res['failed_papers']} | "
            f"Elements: {parse_res['total_elements']}, "
            f"Hierarchy edges: {parse_res['total_hierarchy']}, "
            f"BibTeX entries: {parse_res['total_bibtex']}"
        )

    # Stage 2: Export Parquet tables
    if stage in ("all", "export"):
        print("\n--- [Stage 2/2] Exporting Standardized Parquet Tables ---")
        t0 = time.perf_counter()
        exp_res = export_parquet_tables(
            student_id=STUDENT_ID,
            intermediate_dir=INTERMEDIATE_DIR,
            output_dir=PROCESSED_DIR,
            arxiv_ids=arxiv_ids,
        )
        print(
            f"Export completed in {time.perf_counter() - t0:.2f}s | "
            f"Elements: {exp_res['elements_count']}, "
            f"Hierarchy: {exp_res['hierarchy_count']}, "
            f"BibTeX: {exp_res['bibtex_count']}"
        )
        print(f"Output files:\n  {exp_res['elements_path']}\n  {exp_res['hierarchy_path']}\n  {exp_res['bibtex_path']}")

    total_time = time.perf_counter() - t_start
    ram_end = _get_ram_usage_mb()
    print("\n=== Pipeline Execution Summary ===")
    print(f"Total Wall Time: {total_time:.2f} seconds")
    print(f"RAM Usage: Initial = {ram_start:.1f} MB, Final = {ram_end:.1f} MB")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=f"LaTeX parsing pipeline for student {STUDENT_ID}"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "parse", "export"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index offset within assigned papers (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only N papers from offset (0 for all remaining)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-parsing of papers even if cached in intermediate/",
    )

    args = parser.parse_args()
    run_pipeline(
        stage=args.stage,
        limit=args.limit,
        offset=args.offset,
        force=args.force,
    )


if __name__ == "__main__":
    main()
