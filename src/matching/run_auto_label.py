"""Run automatic reference labeling and report pool coverage and manual metrics."""
from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.matching.auto_matcher import MATCHED, NO_MATCH, auto_label


ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_CONFIG_PATH = ROOT / "config" / "aXirPaperID"
SEMANTIC_SCHOLAR_DIR = ROOT / "data" / "raw" / "semantic_scholar"
AUTO_LABEL_DIR = ROOT / "data" / "processed" / "labels" / "auto"
MANUAL_LABEL_DIR = ROOT / "data" / "processed" / "labels" / "manual"

_MANUAL_COLUMNS = {"arxiv_id", "reference_index", "label", "matched_arxiv_id"}
_LABEL_COLUMNS = [
    "arxiv_id", "reference_index", "ref_arxiv_id", "ref_doi", "ref_title",
    "ref_authors", "ref_year", "ref_submission_date", "ref_s2_id", "label",
    "matched_arxiv_id", "match_method",
]


def _configured_assignments(path: Path) -> List[dict]:
    assignments = []
    seen = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "member_name", "start_month", "start_id", "end_month", "end_id"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            start_month = row["start_month"].replace("-", "")
            end_month = row["end_month"].replace("-", "")
            if start_month != end_month or len(start_month) != 6:
                raise ValueError(
                    f"Invalid or cross-month assignment in {path}:{line_number}"
                )
            try:
                start_id, end_id = int(row["start_id"]), int(row["end_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid ID range in {path}:{line_number}"
                ) from error
            if start_id < 0 or end_id < start_id or end_id > 99999:
                raise ValueError(f"Invalid ID range in {path}:{line_number}")
            prefix = start_month[2:]
            for number in range(start_id, end_id + 1):
                arxiv_id = f"{prefix}.{number:05d}"
                if arxiv_id in seen:
                    raise ValueError(f"Overlapping assignment for {arxiv_id}")
                seen.add(arxiv_id)
                assignments.append({
                    "arxiv_id": arxiv_id,
                    "member_name": row["member_name"],
                    "assignment_month": row["start_month"],
                })
    if not assignments:
        raise ValueError(f"No paper IDs found in assignment config: {path}")
    return assignments


def _normalize_id(value: object) -> str:
    identifier = str(value or "").strip().casefold()
    return re.sub(r"v\d+$", "", identifier)


def _label_output_path(output_dir: Path, arxiv_id: str) -> Path:
    match = re.fullmatch(r"(?P<month>\d{4})\.(?P<number>\d{5})", arxiv_id)
    if not match:
        raise ValueError(f"Invalid arXiv ID for label output: {arxiv_id!r}")
    return output_dir / match.group("month") / f"{match.group('number')}.csv"


def _read_manual_labels(path: Optional[Path]) -> Tuple[List[dict], str]:
    if path is None:
        path = MANUAL_LABEL_DIR
    path = Path(path)
    if not path.exists():
        if path == MANUAL_LABEL_DIR:
            return [], "not_available"
        raise FileNotFoundError(f"Manual labels not found: {path}")

    files = [path] if path.is_file() else sorted(path.glob("*.csv"))
    if not files:
        return [], "not_available"

    labels = []
    seen = set()
    for label_file in files:
        with label_file.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or [])
            missing = _MANUAL_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"{label_file} is missing columns: {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                arxiv_id = _normalize_id(row.get("arxiv_id"))
                try:
                    reference_index = int(row["reference_index"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid reference_index in {label_file}:{line_number}"
                    ) from error
                label = str(row.get("label") or "").strip()
                if label not in {MATCHED, NO_MATCH}:
                    raise ValueError(
                        f"Invalid label in {label_file}:{line_number}: {label!r}"
                    )
                matched_arxiv_id = _normalize_id(row.get("matched_arxiv_id"))
                if label == MATCHED and not matched_arxiv_id:
                    raise ValueError(
                        f"Matched manual label has no matched_arxiv_id in "
                        f"{label_file}:{line_number}"
                    )
                if label == NO_MATCH and matched_arxiv_id:
                    raise ValueError(
                        f"No-match manual label has matched_arxiv_id in "
                        f"{label_file}:{line_number}"
                    )
                key = (arxiv_id, reference_index)
                if not arxiv_id or reference_index < 0:
                    raise ValueError(
                        f"Invalid manual reference key in {label_file}:{line_number}"
                    )
                if key in seen:
                    raise ValueError(
                        f"Duplicate manual label for {arxiv_id} reference "
                        f"{reference_index}"
                    )
                seen.add(key)
                labels.append({
                    "arxiv_id": arxiv_id,
                    "reference_index": reference_index,
                    "label": label,
                    "matched_arxiv_id": matched_arxiv_id,
                })
    return labels, "available"


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _evaluate(
    manual_labels: Sequence[Mapping[str, object]],
    predictions: Mapping[Tuple[str, int], Mapping[str, object]],
    manual_status: str,
) -> List[dict]:
    if manual_status != "available":
        return [{
            "arxiv_id": "all",
            "status": "not_available",
            "manual_entry_count": 0,
            "evaluated_entry_count": 0,
            "unmatched_manual_count": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": None,
            "recall": None,
            "f1": None,
        }]

    grouped = defaultdict(list)
    for label in manual_labels:
        grouped[str(label["arxiv_id"])].append(label)
    scopes = [("all", manual_labels)] + sorted(grouped.items())
    report = []
    for scope, labels in scopes:
        tp = fp = fn = evaluated = unmatched = 0
        for truth in labels:
            key = (str(truth["arxiv_id"]), int(truth["reference_index"]))
            prediction = predictions.get(key)
            if prediction is None:
                unmatched += 1
                if truth["matched_arxiv_id"]:
                    fn += 1
                continue
            evaluated += 1
            truth_id = str(truth["matched_arxiv_id"])
            predicted_id = _normalize_id(prediction.get("matched_arxiv_id"))
            if predicted_id and predicted_id == truth_id:
                tp += 1
            else:
                if predicted_id:
                    fp += 1
                if truth_id:
                    fn += 1

        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1_denominator = 2 * tp + fp + fn
        f1 = _ratio(2 * tp, f1_denominator)
        report.append({
            "arxiv_id": scope,
            "status": "available",
            "manual_entry_count": len(labels),
            "evaluated_entry_count": evaluated,
            "unmatched_manual_count": unmatched,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return report


def _write_csv(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    columns: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_auto_label(
    arxiv_ids: Optional[Iterable[str]] = None,
    *,
    config_path: Path = ASSIGNMENT_CONFIG_PATH,
    source_dir: Path = SEMANTIC_SCHOLAR_DIR,
    output_dir: Path = AUTO_LABEL_DIR,
    manual_labels_path: Optional[Path] = None,
    coverage_report_path: Optional[Path] = None,
    evaluation_report_path: Optional[Path] = None,
) -> Tuple[List[dict], List[dict]]:
    """Label the available pool and write per-paper results and summary reports."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    coverage_report_path = Path(
        coverage_report_path or output_dir / "coverage_report.csv"
    )
    evaluation_report_path = Path(
        evaluation_report_path or output_dir / "evaluation_report.csv"
    )
    assignments = (
        [
            {
                "arxiv_id": str(identifier).strip(),
                "member_name": "",
                "assignment_month": "",
            }
            for identifier in sorted({str(value).strip() for value in arxiv_ids})
        ]
        if arxiv_ids is not None
        else _configured_assignments(Path(config_path))
    )

    predictions: Dict[Tuple[str, int], Mapping[str, object]] = {}
    coverage = []
    available_paper_count = 0
    for assignment in assignments:
        arxiv_id = assignment["arxiv_id"]
        source_path = source_dir / f"{arxiv_id}.json"
        references_available = source_path.is_file()
        rows = auto_label(arxiv_id) if references_available else []
        if references_available:
            available_paper_count += 1
        output_path = _label_output_path(output_dir, arxiv_id)
        _write_csv(rows, output_path, columns=_LABEL_COLUMNS)
        legacy_path = output_dir / f"{arxiv_id}.csv"
        if legacy_path.is_file():
            legacy_path.unlink()
        matched_count = sum(row["label"] == MATCHED for row in rows)
        no_match_count = len(rows) - matched_count
        coverage.append({
            "arxiv_id": arxiv_id,
            "member_name": assignment["member_name"],
            "assignment_month": assignment["assignment_month"],
            "source_status": (
                "available" if references_available else "missing_references"
            ),
            "entry_count": len(rows),
            "matched_count": matched_count,
            "no_match_count": no_match_count,
            "matched_rate": _ratio(matched_count, len(rows)),
            "no_match_rate": _ratio(no_match_count, len(rows)),
        })
        for row in rows:
            key = (str(row["arxiv_id"]), int(row["reference_index"]))
            if key in predictions:
                raise ValueError(
                    f"Duplicate automatic prediction for "
                    f"{key[0]} reference {key[1]}"
                )
            predictions[key] = row

    total_entries = sum(row["entry_count"] for row in coverage)
    total_matched = sum(row["matched_count"] for row in coverage)
    total_no_match = sum(row["no_match_count"] for row in coverage)
    coverage.append({
        "arxiv_id": "all",
        "member_name": "all",
        "assignment_month": "all",
        "source_status": "summary",
        "assigned_paper_count": len(assignments),
        "papers_with_references": available_paper_count,
        "papers_missing_references": len(assignments) - available_paper_count,
        "entry_count": total_entries,
        "matched_count": total_matched,
        "no_match_count": total_no_match,
        "matched_rate": _ratio(total_matched, total_entries),
        "no_match_rate": _ratio(total_no_match, total_entries),
    })

    manual_labels, manual_status = _read_manual_labels(manual_labels_path)
    evaluation = _evaluate(manual_labels, predictions, manual_status)
    _write_csv(coverage, coverage_report_path)
    _write_csv(evaluation, evaluation_report_path)
    return coverage, evaluation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label all Semantic Scholar references and report coverage."
    )
    parser.add_argument(
        "--config", type=Path, default=ASSIGNMENT_CONFIG_PATH,
        help="Assignment config used to enumerate all papers.",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=SEMANTIC_SCHOLAR_DIR,
        help="Directory containing <arxiv_id>.json input files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=AUTO_LABEL_DIR,
        help="Directory for per-paper labels and summary reports.",
    )
    parser.add_argument(
        "--manual-labels", type=Path, default=None,
        help=(
            "Manual labels CSV or directory (defaults to the manual "
            "labels directory)."
        ),
    )
    args = parser.parse_args(argv)
    coverage, evaluation = run_auto_label(
        config_path=args.config,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        manual_labels_path=args.manual_labels,
    )
    pool = coverage[-1]
    print(
        f"assigned papers: {pool['assigned_paper_count']}; "
        f"with references: {pool['papers_with_references']}; "
        f"missing references: {pool['papers_missing_references']}; "
        f"entries: {pool['entry_count']}; "
        f"matched: {pool['matched_count']} ({pool['matched_rate']}); "
        f"no_match: {pool['no_match_count']} ({pool['no_match_rate']})"
    )
    overall_evaluation = evaluation[0]
    if overall_evaluation["status"] == "not_available":
        print("manual evaluation: unavailable (no manual labels found)")
    else:
        print(
            "manual evaluation: "
            f"precision={overall_evaluation['precision']}, "
            f"recall={overall_evaluation['recall']}, "
            f"f1={overall_evaluation['f1']}, "
            f"evaluated={overall_evaluation['evaluated_entry_count']}/"
            f"{overall_evaluation['manual_entry_count']}"
        )
    print(f"coverage report: {args.output_dir / 'coverage_report.csv'}")
    print(f"evaluation report: {args.output_dir / 'evaluation_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
