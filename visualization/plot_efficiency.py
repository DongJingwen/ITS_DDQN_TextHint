"""Plot learning efficiency curves from a CSV training log."""

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

try:
    from visualization.log_utils import find_default_training_log
except ModuleNotFoundError:
    from log_utils import find_default_training_log


def _latest_log_path() -> Path:
    return find_default_training_log(LOG_DIR)


def _read_columns(csv_path: Path, required_columns: list[str]) -> dict[str, list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        data = {column: [] for column in required_columns}
        for row in reader:
            for column in required_columns:
                data[column].append(float(row[column]))

    return data


def plot_attempts_curve(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save the average attempts curve figure."""
    csv_path = Path(csv_path) if csv_path else _latest_log_path()
    print(f"Using training log for attempts curve: {csv_path}")
    output_path = Path(output_path) if output_path else FIGURE_DIR / "average_attempts_curve.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = _read_columns(csv_path, ["episode", "average_attempts"])

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(data["episode"], data["average_attempts"], linewidth=2)
    plt.title("Average Attempts per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Attempts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_response_time_curve(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save the average response time curve figure."""
    csv_path = Path(csv_path) if csv_path else _latest_log_path()
    print(f"Using training log for response time curve: {csv_path}")
    output_path = (
        Path(output_path) if output_path else FIGURE_DIR / "average_response_time_curve.png"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = _read_columns(csv_path, ["episode", "average_response_time"])

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(data["episode"], data["average_response_time"], linewidth=2)
    plt.title("Average Response Time per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Response Time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tutoring efficiency curves.")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--attempts-output", type=Path, default=None)
    parser.add_argument("--response-time-output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attempts_path = plot_attempts_curve(args.log, args.attempts_output)
    response_time_path = plot_response_time_curve(args.log, args.response_time_output)
    print(f"Saved average attempts curve: {attempts_path}")
    print(f"Saved average response time curve: {response_time_path}")
