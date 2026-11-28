import csv
import json

import pandas as pd

import src.matching.run_auto_label as runner


def test_run_auto_label_writes_per_paper_coverage_and_exact_match_metrics(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "2603.10580.json").write_text(
        json.dumps({"arxiv_id": "2603.10580", "references": []}),
        encoding="utf-8",
    )
    (source_dir / "2603.10581.json").write_text(
        json.dumps({"arxiv_id": "2603.10581", "references": []}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    config_path = tmp_path / "assignments.csv"
    config_path.write_text(
        "student_id,member_name,start_month,start_id,end_month,end_id\n"
        "1,Member,2026-03,10580,2026-03,10581\n",
        encoding="utf-8",
    )
    manual_file = tmp_path / "manual.csv"
    pd.DataFrame([
        {
            "arxiv_id": "2603.10580",
            "reference_index": 0,
            "label": "matched",
            "matched_arxiv_id": "2501.00001",
        },
        {
            "arxiv_id": "2603.10580",
            "reference_index": 1,
            "label": "matched",
            "matched_arxiv_id": "2501.00002",
        },
        {
            "arxiv_id": "2603.10580",
            "reference_index": 2,
            "label": "no_match",
            "matched_arxiv_id": "",
        },
        {
            "arxiv_id": "2603.10580",
            "reference_index": 3,
            "label": "matched",
            "matched_arxiv_id": "2501.00004",
        },
    ]).to_csv(manual_file, index=False)
    generated = {
        "2603.10580": [
            {"arxiv_id": "2603.10580", "reference_index": 0,
             "label": "matched", "matched_arxiv_id": "2501.00001"},
            {"arxiv_id": "2603.10580", "reference_index": 1,
             "label": "matched", "matched_arxiv_id": "2501.00003"},
            {"arxiv_id": "2603.10580", "reference_index": 2,
             "label": "matched", "matched_arxiv_id": "2501.00004"},
        ],
        "2603.10581": [
            {"arxiv_id": "2603.10581", "reference_index": 0,
             "label": "no_match", "matched_arxiv_id": None},
        ],
    }
    monkeypatch.setattr(runner, "auto_label", lambda arxiv_id: generated[arxiv_id])
    monkeypatch.setattr(runner, "MANUAL_LABEL_DIR", tmp_path / "missing_manual")

    coverage, evaluation = runner.run_auto_label(
        config_path=config_path,
        source_dir=source_dir,
        output_dir=output_dir,
        manual_labels_path=manual_file,
    )

    assert coverage[-1] == {
        "arxiv_id": "all",
        "member_name": "all",
        "assignment_month": "all",
        "source_status": "summary",
        "assigned_paper_count": 2,
        "papers_with_references": 2,
        "papers_missing_references": 0,
        "entry_count": 4,
        "matched_count": 3,
        "no_match_count": 1,
        "matched_rate": 0.75,
        "no_match_rate": 0.25,
    }
    overall = evaluation[0]
    assert overall["manual_entry_count"] == 4
    assert overall["evaluated_entry_count"] == 3
    assert overall["unmatched_manual_count"] == 1
    assert overall["true_positive"] == 1
    assert overall["false_positive"] == 2
    assert overall["false_negative"] == 2
    assert overall["precision"] == 1 / 3
    assert overall["recall"] == 1 / 3
    assert overall["f1"] == 1 / 3
    assert (output_dir / "2603" / "10580.csv").exists()
    with (output_dir / "2603" / "10580.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        assert len(list(csv.DictReader(file))) == 3
    assert pd.read_csv(output_dir / "coverage_report.csv").iloc[-1]["entry_count"] == 4
    assert pd.read_csv(output_dir / "evaluation_report.csv").iloc[0][
        "unmatched_manual_count"
    ] == 1


def test_run_auto_label_reports_manual_evaluation_unavailable(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "2603.10580.json").write_text(
        json.dumps({"arxiv_id": "2603.10580", "references": []}),
        encoding="utf-8",
    )
    config_path = tmp_path / "assignments.csv"
    config_path.write_text(
        "student_id,member_name,start_month,start_id,end_month,end_id\n"
        "1,Member,2026-03,10580,2026-03,10581\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "auto_label",
        lambda arxiv_id: [
            {"arxiv_id": arxiv_id, "reference_index": 0, "label": "no_match",
             "matched_arxiv_id": None}
        ],
    )
    monkeypatch.setattr(runner, "MANUAL_LABEL_DIR", tmp_path / "missing_manual")

    _, evaluation = runner.run_auto_label(
        config_path=config_path,
        source_dir=source_dir,
        output_dir=tmp_path / "output",
    )

    assert evaluation == [{
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


def test_run_auto_label_creates_config_placeholders_for_missing_references(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "2603.10580.json").write_text(
        json.dumps({
            "arxiv_id": "2603.10580",
            "references": [{
                "title": "No candidate",
                "externalIds": {},
            }],
        }),
        encoding="utf-8",
    )
    config_path = tmp_path / "assignments.csv"
    config_path.write_text(
        "student_id,member_name,start_month,start_id,end_month,end_id\n"
        "1,Hung,2026-03,10580,2026-03,10582\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "auto_label",
        lambda arxiv_id: [{
            "arxiv_id": arxiv_id,
            "reference_index": 0,
            "label": "no_match",
            "matched_arxiv_id": None,
        }],
    )
    monkeypatch.setattr(runner, "MANUAL_LABEL_DIR", tmp_path / "missing_manual")
    output_dir = tmp_path / "output"

    coverage, _ = runner.run_auto_label(
        config_path=config_path,
        source_dir=source_dir,
        output_dir=output_dir,
    )

    assert len(coverage) == 4
    assert [row["source_status"] for row in coverage[:-1]] == [
        "available", "missing_references", "missing_references"
    ]
    assert coverage[-1]["assigned_paper_count"] == 3
    assert coverage[-1]["papers_with_references"] == 1
    assert coverage[-1]["papers_missing_references"] == 2
    assert len(list(output_dir.rglob("*.csv"))) == 5
    for arxiv_id in ("2603.10581", "2603.10582"):
        prefix, number = arxiv_id.split(".")
        with (output_dir / prefix / f"{number}.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            assert list(csv.DictReader(file)) == []


def test_label_output_path_uses_arxiv_month_folder(tmp_path):
    assert runner._label_output_path(
        tmp_path, "2603.10580"
    ) == tmp_path / "2603" / "10580.csv"
