"""Publication-level train / valid / test split for Milestone 3.

Reads labels_manual.csv and labels_auto.csv and writes split.json:
  test  = k manual + k automatic publications
  valid = k manual + k automatic publications
  train = every remaining labelled publication (k = number of members)

Usage:
    python -m src.matching.split_data --members 1 --seed 42
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.matching.auto_labeler import AUTO_LABELS_FILE
from src.matching.build_candidates import CSV_ENCODING, MANUAL_LABELS_FILE
from src.matching.select_publications import DEFAULT_OUTPUT_DIR

SPLIT_FILE = "split.json"
PARTITIONS = ("train", "valid", "test")
ORIGINS = ("manual", "auto")

# arxiv_id -> {citation_key: true reference arXiv ID, "" when no reference matches}
Labels = Dict[str, Dict[str, str]]


def load_labels(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Tuple[Labels, Dict[str, str]]:
    """Load manual and automatic labels.

    Returns:
        (labels, source) where source maps each publication to "manual" or "auto".
    """
    labels: Labels = {}
    source: Dict[str, str] = {}
    # Automatic first, manual last: a manual label always overrides an automatic one
    for file_name, origin in ((AUTO_LABELS_FILE, "auto"), (MANUAL_LABELS_FILE, "manual")):
        path = output_dir / file_name
        if not path.exists():
            continue
        with path.open("r", encoding=CSV_ENCODING, newline="") as f:
            for row in csv.DictReader(f):
                aid = row["arxiv_id"]
                truth = (row.get("groundtruth_arxiv_id") or "").strip()
                labels.setdefault(aid, {})[row["citation_key"]] = truth
                source[aid] = origin
    return labels, source


def _positives(pub_labels: Dict[str, str]) -> int:
    """Number of labelled entries that have a true reference."""
    return sum(1 for truth in pub_labels.values() if truth)


def make_split(
    labels: Labels,
    source: Dict[str, str],
    members: int = 1,
    min_manual_entries: int = 20,
    min_positives: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """Split labelled publications into train / valid / test.

    Manual publications qualify with >= min_manual_entries labelled entries. Publications
    with at least min_positives positive labels are preferred for valid/test, so that the
    MRR is computed on a meaningful number of positives.
    """
    manual = sorted(a for a, s in source.items() if s == "manual" and len(labels[a]) >= min_manual_entries)
    auto = sorted(a for a, s in source.items() if s == "auto")
    needed = 2 * members + 1
    if len(manual) < needed or len(auto) < needed:
        raise ValueError(
            f"Need >= {needed} manual and >= {needed} automatic publications, "
            f"found {len(manual)} manual / {len(auto)} automatic."
        )

    rng = random.Random(seed)
    split: Dict[str, Dict[str, List[str]]] = {p: {} for p in PARTITIONS}
    for origin, pool in (("manual", manual), ("auto", auto)):
        pool = list(pool)
        rng.shuffle(pool)
        # Stable sort: eligible publications first, random order inside each group
        pool.sort(key=lambda a: _positives(labels[a]) < min_positives)
        split["test"][origin] = pool[:members]
        split["valid"][origin] = pool[members : 2 * members]
        split["train"][origin] = sorted(pool[2 * members :])
    return split


def save_split(split: Dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write split.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SPLIT_FILE
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return path


def load_split(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Dict[str, List[str]]]:
    """Read split.json."""
    return json.loads((output_dir / SPLIT_FILE).read_text(encoding="utf-8"))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Publication-level train/valid/test split")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--members", type=int, default=1, help="Group size k (k publications per set and origin)")
    parser.add_argument("--min-manual-entries", type=int, default=20)
    parser.add_argument("--min-positives", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels, source = load_labels(args.output_dir)
    split = make_split(labels, source, args.members, args.min_manual_entries, args.min_positives, args.seed)
    path = save_split(split, args.output_dir)

    print(f"Split written to {path}")
    for part in PARTITIONS:
        for origin in ORIGINS:
            pubs = split[part][origin]
            n_entries = sum(len(labels[a]) for a in pubs)
            n_pos = sum(_positives(labels[a]) for a in pubs)
            print(f"  {part:5s} {origin:6s} publications={len(pubs):4d} labelled_entries={n_entries:6d} positives={n_pos:6d}")


if __name__ == "__main__":
    main()