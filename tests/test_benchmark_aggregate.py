from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.utils.benchmark import FIELDS
from src.utils.benchmark_aggregate import aggregate_benchmarks, main


def _write_fake_benchmarks(csv_path: Path) -> None:
    rows = [
        {
            "timestamp_utc": "2026-11-12T08:00:00+00:00",
            "label": "m1_crawl_part1",
            "n_items": 1500,
            "wall_time_s": 100.0,
            "time_per_item_s": 0.0667,
            "peak_ram_mib": 120.0,
            "avg_ram_mib": 60.0,
            "output_size_mib": 300.0,
            "peak_disk_mib": 310.0,
            "n_samples": 100,
            "returncode": 0,
            "platform": "linux, 2 CPU, 12.7 GiB RAM",
        },
        {
            "timestamp_utc": "2026-11-12T10:00:00+00:00",
            "label": "m1_crawl_part2",
            "n_items": 2000,
            "wall_time_s": 250.5,
            "time_per_item_s": 0.1253,
            "peak_ram_mib": 185.4,
            "avg_ram_mib": 90.0,
            "output_size_mib": 720.0,
            "peak_disk_mib": 750.5,
            "n_samples": 300,
            "returncode": 0,
            "platform": "linux, 2 CPU, 12.7 GiB RAM",
        },
        {
            "timestamp_utc": "2026-11-12T12:00:00+00:00",
            "label": "m1_crawl_part3",
            "n_items": 1500,
            "wall_time_s": 149.5,
            "time_per_item_s": 0.0997,
            "peak_ram_mib": 150.2,
            "avg_ram_mib": 75.0,
            "output_size_mib": 1150.25,
            "peak_disk_mib": 1180.0,
            "n_samples": 100,
            "returncode": 0,
            "platform": "linux, 2 CPU, 12.7 GiB RAM",
        },
        {
            "timestamp_utc": "2026-11-12T14:00:00+00:00",
            "label": "m2_parser",
            "n_items": 5000,
            "wall_time_s": 999.0,
            "time_per_item_s": 0.1998,
            "peak_ram_mib": 4096.0,
            "avg_ram_mib": 500.0,
            "output_size_mib": 900.0,
            "peak_disk_mib": 900.0,
            "n_samples": 500,
            "returncode": 0,
            "platform": "linux, 2 CPU, 12.7 GiB RAM",
        },
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_three_sessions(tmp_path: Path):
    csv_path = tmp_path / "benchmarks.csv"
    _write_fake_benchmarks(csv_path)

    result = aggregate_benchmarks(csv_path=csv_path, prefix="m1_crawl")

    assert result["sessions"] == 3
    assert result["n_items"] == 5000
    assert result["wall_time_s"] == 500.0
    assert result["time_per_item_s"] == 0.1
    assert result["peak_ram_mib"] == 185.4
    assert result["avg_ram_mib"] == 81.0
    assert result["output_size_mib"] == 1150.25
    assert result["peak_disk_mib"] == 1180.0
    assert result["n_samples"] == 500


def test_cli_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    csv_path = tmp_path / "benchmarks.csv"
    _write_fake_benchmarks(csv_path)

    exit_code = main(["--csv", str(csv_path), "--prefix", "m1_crawl"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "sessions: 3" in captured
    assert "wall_time_s: 500.0" in captured
    assert "peak_ram_mib: 185.4" in captured
    assert "avg_ram_mib: 81.0" in captured
    assert "peak_disk_mib: 1180.0" in captured