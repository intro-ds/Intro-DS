"""Check a published dataset with load_dataset(..., streaming=True) for all six configs.

Usage:
    python -m src.matching.verify_hf_dataset --repo-id <username>/<dataset-name>
"""

import argparse
import sys

from datasets import load_dataset

from src.matching.export_hf_dataset import EXPECTED_COLUMNS, TABLES


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Verify the six dataset configs in streaming mode")
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    failures = 0
    for table in TABLES:
        try:
            stream = load_dataset(args.repo_id, table, split="train", streaming=True)
            first = next(iter(stream), None)
            if first is None:
                print(f"[WARN] {table}: config loads but has no rows")
                failures += 1
                continue
            columns = list(first.keys())
            ok = columns == EXPECTED_COLUMNS[table]
            failures += 0 if ok else 1
            print(f"[{'OK' if ok else 'FAIL'}] {table}: columns={columns}")
        except Exception as exc:  # report every config instead of stopping at the first error
            failures += 1
            print(f"[FAIL] {table}: {type(exc).__name__}: {exc}")

    print("\nALL 6 CONFIGS OK" if failures == 0 else f"\n{failures} problem(s) found")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()