"""Remove non-reference @CONTROL entries from already-parsed intermediate files.

Shortcut that avoids re-parsing the whole corpus after the parse_bib_file fix:
entries whose entry_type is "control" are dropped from the "bibtex" list of every
data/intermediate/<arxiv_id>/parsed.json. Re-run the parquet export afterwards.

Usage:
    python -m src.utils.clean_control_entries [--intermediate-dir data/intermediate] [--dry-run]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

CONTROL_TYPES = {"control"}


def clean_intermediate(intermediate_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    """Drop @CONTROL entries from every parsed.json and report the counts.

    Args:
        intermediate_dir: Directory containing <arxiv_id>/parsed.json files.
        dry_run: If True, only count; do not modify any file.

    Returns:
        Dict with papers scanned, papers changed and entries removed.
    """
    scanned = changed = removed = 0
    for path in sorted(intermediate_dir.glob("*/parsed.json")):
        scanned += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        entries = data.get("bibtex") or []
        kept = [e for e in entries if str(e.get("entry_type", "")).lower() not in CONTROL_TYPES]
        n_removed = len(entries) - len(kept)
        if n_removed == 0:
            continue

        changed += 1
        removed += n_removed
        if dry_run:
            continue

        data["bibtex"] = kept
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)  # atomic: never leaves a half-written file

    return {"papers_scanned": scanned, "papers_changed": changed, "entries_removed": removed}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Remove @CONTROL entries from parsed.json files")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    parser.add_argument("--dry-run", action="store_true", help="Count only, do not modify files")
    args = parser.parse_args()
    res = clean_intermediate(args.intermediate_dir, args.dry_run)
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}papers_scanned={res['papers_scanned']} papers_changed={res['papers_changed']} "
          f"entries_removed={res['entries_removed']}")


if __name__ == "__main__":
    main()