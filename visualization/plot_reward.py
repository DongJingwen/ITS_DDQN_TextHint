"""Plot training reward curve from a CSV training log."""

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


def _latest_log_path() -> Path:
    log_files = sorted(LOG_DIR.glob("training_log_*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {LOG_DIR}")
    return log_files[-1]


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


def plot_reward_curve(csv_path: str | Path | None = None, output_path: str | Path | None = None) -> Path:
    """Generate and save the reward curve figure."""
    csv_path = Path(csv_path) if csv_path else _latest_log_path()
    output_path = Path(output_path) if output_path else FIGURE_DIR / "reward_curve.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = _read_columns(csv_path, ["episode", "total_reward"])

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(data["episode"], data["total_reward"], linewidth=2)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DDQN training reward curve.")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved_path = plot_reward_curve(args.log, args.output)
    print(f"Saved reward curve: {saved_path}")
