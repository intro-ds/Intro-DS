"""Import metadata exported by the history crawler into the corpus layout.

Reads:
  * arxiv_history_<YYMM>.jsonl              -> {"base_id", "versions": [...]}
  * arxiv_metadata_<YYMM>_all_versions.jsonl -> one record per version

Writes data/raw/corpus/<base_id>/metadata.json in the same schema produced by
entry_discovery.serialize_arxiv_result, so source_downloader and
dataset_exporter can consume it unchanged.

Usage:
    python -m src.crawling.import_history_metadata \
        --history data/raw/arxiv_history_2511.jsonl \
        --metadata data/raw/arxiv_metadata_2511_all_versions.jsonl
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, List

DEFAULT_CORPUS_DIR = Path("data/raw/corpus")


def _version_number(version_id: str) -> int:
    """Return the integer N of an id such as '2511.16712v2'."""
    return int(re.search(r"v(\d+)$", version_id).group(1))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON-lines file, tolerating CRLF line endings and blank lines."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def import_metadata(
    history_path: Path,
    metadata_path: Path,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    force: bool = False,
) -> Dict[str, Any]:
    """Convert history + per-version metadata into corpus/<id>/metadata.json."""
    history = {r["base_id"]: r["versions"] for r in _read_jsonl(history_path)}

    per_version: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for rec in _read_jsonl(metadata_path):
        base = re.sub(r"v\d+$", "", rec["id"])
        per_version[base][_version_number(rec["id"])] = rec

    written, skipped = 0, 0
    incomplete: List[str] = []

    for base_id, versions in history.items():
        nums = sorted(_version_number(v) for v in versions)
        recs = per_version.get(base_id, {})
        if not nums or any(n not in recs for n in nums):
            incomplete.append(base_id)
            continue

        out_file = corpus_dir / base_id / "metadata.json"
        if out_file.exists() and not force:
            skipped += 1
            continue

        first, last = recs[nums[0]], recs[nums[-1]]
        data = {
            "arxiv_id": base_id,
            "title": (last.get("title") or "").strip(),
            "authors": list(last.get("authors") or []),
            "submission_date": first.get("published"),
            # one revision date per version >= 2 (the 'updated' stamp of that version)
            "revised_dates": [recs[n]["updated"] for n in nums if n > 1],
            "abstract": (last.get("abstract") or "").strip(),
            "venue": last.get("journal_ref") or None,
            "primary_category": last.get("primary_category"),
            "categories": list(last.get("categories") or []),
            "doi": last.get("doi"),
            "versions": [f"{base_id}v{n}" for n in nums],
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        written += 1

    return {
        "total": len(history),
        "written": written,
        "skipped": skipped,
        "incomplete": incomplete,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Import history metadata into corpus")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    res = import_metadata(args.history, args.metadata, args.corpus_dir, args.force)
    print(
        f"total={res['total']} written={res['written']} skipped={res['skipped']} "
        f"incomplete={len(res['incomplete'])}"
    )
    if res["incomplete"]:
        print("Incomplete ids (history/metadata mismatch):", res["incomplete"][:20])


if __name__ == "__main__":
    main()