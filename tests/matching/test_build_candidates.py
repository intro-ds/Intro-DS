"""Unit tests for build_candidates."""

import csv

from src.matching.build_candidates import collect_manual_labels, generate_candidates

AID = "2512.00001"


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_generate_writes_candidates_and_reference_list(workspace, matched_publication) -> None:
    matched_publication(AID)
    res = generate_candidates(AID, workspace.intermediate_dir, workspace.corpus_dir, workspace.output_dir, max_entries=0)
    assert res["status"] == "written"

    rows = _read(workspace.output_dir / "manual_labels" / f"{AID}.csv")
    assert len(rows) == 5
    assert rows[0]["cand1_id"] == "1706.03762"  # best suggestion is the true reference
    assert rows[0]["label"] == ""
    assert (workspace.output_dir / "manual_labels" / "references" / f"{AID}.csv").exists()


def test_generate_does_not_overwrite_manual_work(workspace, matched_publication) -> None:
    matched_publication(AID)
    args = (AID, workspace.intermediate_dir, workspace.corpus_dir, workspace.output_dir)
    generate_candidates(*args, max_entries=0)
    assert generate_candidates(*args, max_entries=0)["status"] == "skipped_exists"
    assert generate_candidates(*args, max_entries=0, force=True)["status"] == "written"


def test_generate_samples_max_entries(workspace, matched_publication) -> None:
    matched_publication(AID)
    generate_candidates(AID, workspace.intermediate_dir, workspace.corpus_dir, workspace.output_dir, max_entries=3)
    assert len(_read(workspace.output_dir / "manual_labels" / f"{AID}.csv")) == 3


def test_collect_validates_and_counts(workspace, matched_publication) -> None:
    matched_publication(AID)
    generate_candidates(AID, workspace.intermediate_dir, workspace.corpus_dir, workspace.output_dir, max_entries=0)
    path = workspace.output_dir / "manual_labels" / f"{AID}.csv"
    rows = _read(path)
    rows[0]["label"] = "1"                 # rank shortcut
    rows[1]["label"] = "1512.03385"        # explicit arXiv ID (known reference)
    rows[2]["label"] = "NONE"              # no match
    rows[3]["label"] = "9999.99999"        # invalid ID -> error
    # rows[4] left blank -> ignored
    _write(path, rows)

    res = collect_manual_labels(workspace.output_dir, workspace.corpus_dir, members=1)
    assert res["per_publication"][AID] == 3
    assert res["entries_positive"] == 2 and res["entries_none"] == 1
    assert len(res["errors"]) == 1
    assert res["requirement_met"] is False  # far below 5 publications / 100 entries

    out = _read(workspace.output_dir / "labels_manual.csv")
    assert out[0]["groundtruth_arxiv_id"] == rows[0]["cand1_id"] == "1706.03762"
    assert out[1]["groundtruth_arxiv_id"] == "1512.03385"
    assert out[2]["groundtruth_arxiv_id"] == ""


def test_collect_requirement_met(workspace, matched_publication) -> None:
    # Fabricate 5 publications with 20 labelled rows each via direct CSV writing.
    labels_dir = workspace.output_dir / "manual_labels"
    labels_dir.mkdir(parents=True)
    for p in range(5):
        aid = f"2512.0000{p + 1}"
        matched_publication(aid)
        rows = [
            {"arxiv_id": aid, "citation_key": f"k{i}", "cand1_id": "1706.03762", "label": "1"}
            for i in range(20)
        ]
        _write(labels_dir / f"{aid}.csv", rows)
    res = collect_manual_labels(workspace.output_dir, workspace.corpus_dir, members=1)
    assert res["qualifying_publications"] == 5
    assert res["qualifying_entries"] == 100
    assert res["requirement_met"] is True