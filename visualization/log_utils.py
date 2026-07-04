"""Helpers for selecting training logs for visualization."""

from __future__ import annotations

import csv
from pathlib import Path


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def find_default_training_log(log_dir: Path, min_episodes: int = 10) -> Path:
    """Choose the most useful training log for figures.

    Timestamp order is fragile because short smoke tests can be newer than
    thesis-scale runs. By default, prefer the log with the most episodes.
    """
    fixed_path = log_dir / "training_log.csv"
    if fixed_path.exists():
        return fixed_path

    log_files = sorted(log_dir.glob("training_log_*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {log_dir}")

    ranked_logs = [(count_csv_rows(path), path) for path in log_files]
    ranked_logs.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    row_count, selected_path = ranked_logs[0]

    if row_count < min_episodes:
        print(
            "Warning: selected training log has only "
            f"{row_count} episodes, so trend figures may look too linear: {selected_path}"
        )

    return selected_path
