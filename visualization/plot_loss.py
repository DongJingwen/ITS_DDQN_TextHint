"""Plot loss and dynamic epsilon curves from a CSV training log."""

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


def plot_loss_curve(csv_path: str | Path | None = None, output_path: str | Path | None = None) -> Path:
    """Generate and save the average loss curve figure."""
    csv_path = Path(csv_path) if csv_path else _latest_log_path()
    print(f"Using training log for loss curve: {csv_path}")
    output_path = Path(output_path) if output_path else FIGURE_DIR / "loss_curve.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = _read_columns(csv_path, ["episode", "average_loss"])

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(data["episode"], data["average_loss"], linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_epsilon_curve(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save the dynamic epsilon curve figure."""
    csv_path = Path(csv_path) if csv_path else _latest_log_path()
    print(f"Using training log for epsilon curve: {csv_path}")
    output_path = Path(output_path) if output_path else FIGURE_DIR / "dynamic_epsilon_curve.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = _read_columns(csv_path, ["episode", "average_epsilon"])

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(data["episode"], data["average_epsilon"], linewidth=2)
    plt.title("Dynamic Epsilon Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Dynamic Epsilon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DDQN loss and epsilon curves.")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--loss-output", type=Path, default=None)
    parser.add_argument("--epsilon-output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loss_path = plot_loss_curve(args.log, args.loss_output)
    epsilon_path = plot_epsilon_curve(args.log, args.epsilon_output)
    print(f"Saved loss curve: {loss_path}")
    print(f"Saved dynamic epsilon curve: {epsilon_path}")
