from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.utils.benchmark import RESULTS_PATH


def _to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _to_int(value: str | float | int | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def aggregate_benchmarks(
    csv_path: Path = RESULTS_PATH,
    prefix: str | None = None,
    include_failed: bool = False,
) -> dict:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

    matched_rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if prefix and not label.startswith(prefix):
                continue
            if not include_failed and _to_int(row.get("returncode"), 0) != 0:
                continue
            matched_rows.append(row)

    if not matched_rows:
        return {
            "prefix": prefix or "*",
            "sessions": 0,
            "n_items": 0,
            "wall_time_s": 0.0,
            "time_per_item_s": None,
            "peak_ram_mib": 0.0,
            "avg_ram_mib": 0.0,
            "output_size_mib": 0.0,
            "peak_disk_mib": 0.0,
            "n_samples": 0,
        }

    total_wall = sum(_to_float(r.get("wall_time_s")) for r in matched_rows)
    total_items = sum(_to_int(r.get("n_items")) for r in matched_rows)
    peak_ram = max(_to_float(r.get("peak_ram_mib")) for r in matched_rows)
    peak_disk = max(_to_float(r.get("peak_disk_mib")) for r in matched_rows)
    output_size = max(_to_float(r.get("output_size_mib")) for r in matched_rows)

    total_samples = sum(_to_int(r.get("n_samples")) for r in matched_rows)
    if total_samples > 0:
        weighted_avg_ram = (
            sum(
                _to_float(r.get("avg_ram_mib")) * _to_int(r.get("n_samples"))
                for r in matched_rows
            )
            / total_samples
        )
    else:
        weighted_avg_ram = sum(_to_float(r.get("avg_ram_mib")) for r in matched_rows) / len(matched_rows)

    time_per_item = round(total_wall / total_items, 4) if total_items > 0 else None

    return {
        "prefix": prefix or "*",
        "sessions": len(matched_rows),
        "n_items": total_items,
        "wall_time_s": round(total_wall, 3),
        "time_per_item_s": time_per_item,
        "peak_ram_mib": round(peak_ram, 1),
        "avg_ram_mib": round(weighted_avg_ram, 1),
        "output_size_mib": round(output_size, 2),
        "peak_disk_mib": round(peak_disk, 2),
        "n_samples": total_samples,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate multi-session benchmark results.")
    ap.add_argument("--csv", type=Path, default=RESULTS_PATH, help="Path to benchmarks.csv")
    ap.add_argument("--prefix", type=str, default=None, help="Filter rows by label prefix")
    ap.add_argument("--include-failed", action="store_true", help="Include runs with non-zero returncode")
    args = ap.parse_args(argv)

    summary = aggregate_benchmarks(
        csv_path=args.csv,
        prefix=args.prefix,
        include_failed=args.include_failed,
    )
    for key, value in summary.items():
        print(f"{key:>16}: {value if value is not None else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())