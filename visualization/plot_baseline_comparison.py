"""Generate thesis-ready baseline comparison figures."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CSV = PROJECT_ROOT / "outputs" / "logs" / "baseline_comparison.csv"
BASELINE_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "baseline"
BASELINE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
Path(BASELINE_FIGURE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BASELINE_FIGURE_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICY_LABELS = {
    "random": "Random Hint",
    "rule_based": "Rule-based Hint",
    "double_dqn": "Double DQN",
}


def _read_baseline_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Warning: baseline comparison CSV not found: {csv_path}")
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _policy_label(policy_name: str) -> str:
    return POLICY_LABELS.get(policy_name, policy_name)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _plot_metric_bar(
    rows: list[dict[str, Any]],
    metric_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    if not rows:
        print(f"Warning: no data available for {title}.")
        return None

    labels = [_policy_label(row["policy"]) for row in rows]
    values = [_to_float(row.get(metric_column)) for row in rows]

    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Policy")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_average_reward(
    rows: list[dict[str, Any]],
    output_dir: Path = BASELINE_FIGURE_DIR,
) -> Path | None:
    return _plot_metric_bar(
        rows=rows,
        metric_column="average_reward",
        title="Baseline Average Reward Comparison",
        ylabel="Average Reward",
        output_path=output_dir / "baseline_average_reward.png",
    )


def plot_correct_after_hint_rate(
    rows: list[dict[str, Any]],
    output_dir: Path = BASELINE_FIGURE_DIR,
) -> Path | None:
    return _plot_metric_bar(
        rows=rows,
        metric_column="correct_after_hint_rate",
        title="Baseline Correct-after-Hint Rate Comparison",
        ylabel="Correct-after-Hint Rate",
        output_path=output_dir / "baseline_correct_after_hint_rate.png",
    )


def plot_average_attempts(
    rows: list[dict[str, Any]],
    output_dir: Path = BASELINE_FIGURE_DIR,
) -> Path | None:
    return _plot_metric_bar(
        rows=rows,
        metric_column="average_attempts",
        title="Baseline Average Attempts Comparison",
        ylabel="Average Attempts",
        output_path=output_dir / "baseline_average_attempts.png",
    )


def plot_average_response_time(
    rows: list[dict[str, Any]],
    output_dir: Path = BASELINE_FIGURE_DIR,
) -> Path | None:
    return _plot_metric_bar(
        rows=rows,
        metric_column="average_response_time",
        title="Baseline Average Response Time Comparison",
        ylabel="Average Response Time (seconds)",
        output_path=output_dir / "baseline_average_response_time.png",
    )


def plot_hint_distribution(
    rows: list[dict[str, Any]],
    output_dir: Path = BASELINE_FIGURE_DIR,
) -> Path | None:
    if not rows:
        print("Warning: no data available for baseline hint distribution.")
        return None

    labels = [_policy_label(row["policy"]) for row in rows]
    action_0 = [_to_float(row.get("action_0_count")) for row in rows]
    action_1 = [_to_float(row.get("action_1_count")) for row in rows]
    action_2 = [_to_float(row.get("action_2_count")) for row in rows]

    x_positions = range(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 5), dpi=300)
    plt.bar([x - width for x in x_positions], action_0, width, label="Action 0: Strong")
    plt.bar(list(x_positions), action_1, width, label="Action 1: Medium")
    plt.bar([x + width for x in x_positions], action_2, width, label="Action 2: Weak")
    plt.title("Baseline Hint Recommendation Distribution")
    plt.xlabel("Policy")
    plt.ylabel("Action Count")
    plt.xticks(list(x_positions), labels)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "baseline_hint_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_baseline_figures(
    csv_path: str | Path = DEFAULT_BASELINE_CSV,
    output_dir: str | Path = BASELINE_FIGURE_DIR,
) -> list[Path]:
    """Generate all baseline comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_baseline_rows(csv_path)

    if not rows:
        print("Warning: no baseline rows available. No figures were generated.")
        return []

    paths = [
        plot_average_reward(rows, output_dir),
        plot_correct_after_hint_rate(rows, output_dir),
        plot_average_attempts(rows, output_dir),
        plot_average_response_time(rows, output_dir),
        plot_hint_distribution(rows, output_dir),
    ]
    return [path for path in paths if path is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline comparison figures.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_FIGURE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved_paths = generate_baseline_figures(args.csv, args.output_dir)
    for path in saved_paths:
        print(f"Saved baseline figure: {path}")
