"""Plot hint action distribution from a CSV training log."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
Path(FIGURE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_LABELS = [
    "Action 0\nStrong Hint",
    "Action 1\nMedium Hint",
    "Action 2\nWeak Hint",
]


def _latest_log_path() -> Path:
    log_files = sorted(LOG_DIR.glob("training_log_*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {LOG_DIR}")
    return log_files[-1]


def _read_action_counts(csv_path: Path) -> list[int]:
    required_columns = ["action_0_count", "action_1_count", "action_2_count"]
    totals = [0, 0, 0]

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        for row in reader:
            totals[0] += int(float(row["action_0_count"]))
            totals[1] += int(float(row["action_1_count"]))
            totals[2] += int(float(row["action_2_count"]))

    return totals


def plot_hint_distribution(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save the hint action distribution figure."""
    csv_path = Path(csv_path) if csv_path else _latest_log_path()
    output_path = Path(output_path) if output_path else FIGURE_DIR / "hint_action_distribution.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    action_counts = _read_action_counts(csv_path)

    plt.figure(figsize=(8, 5), dpi=150)
    bars = plt.bar(ACTION_LABELS, action_counts)
    plt.title("Hint Action Distribution")
    plt.xlabel("Hint Action")
    plt.ylabel("Total Recommendations")
    plt.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, action_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hint recommendation distribution.")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved_path = plot_hint_distribution(args.log, args.output)
    print(f"Saved hint action distribution: {saved_path}")
