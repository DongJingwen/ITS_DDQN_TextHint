"""Generate thesis-ready figures for real-user experiment results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "logs" / "real_user_metrics.csv"
REAL_USER_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "real_user"
REAL_USER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
Path(REAL_USER_FIGURE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REAL_USER_FIGURE_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_metrics(metrics_csv_path: str | Path) -> list[dict[str, Any]]:
    metrics_csv_path = Path(metrics_csv_path)
    if not metrics_csv_path.exists():
        print(f"Warning: metrics CSV not found: {metrics_csv_path}")
        return []

    with metrics_csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _non_empty_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("student_id")]


def _first_number(rows: list[dict[str, Any]], column: str) -> float | None:
    for row in rows:
        value = _to_float(row.get(column))
        if value is not None:
            return value
    return None


def _save_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    if not labels or not values:
        print(f"Warning: no data available for {title}.")
        return None

    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
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


def plot_pre_post_score_comparison(
    rows: list[dict[str, Any]],
    output_dir: Path = REAL_USER_FIGURE_DIR,
) -> Path | None:
    mean_pretest = _first_number(rows, "mean_pretest_score")
    mean_posttest = _first_number(rows, "mean_posttest_score")
    if mean_pretest is None or mean_posttest is None:
        print("Warning: pre/post score data is unavailable.")
        return None

    return _save_bar_chart(
        labels=["Pre-test", "Post-test"],
        values=[mean_pretest, mean_posttest],
        title="Mean Pre-test and Post-test Score Comparison",
        xlabel="Test Type",
        ylabel="Mean Score",
        output_path=output_dir / "pre_post_score_comparison.png",
    )


def plot_learning_gain_per_student(
    rows: list[dict[str, Any]],
    output_dir: Path = REAL_USER_FIGURE_DIR,
) -> Path | None:
    student_rows = _non_empty_rows(rows)
    labels = []
    values = []
    for row in student_rows:
        learning_gain = _to_float(row.get("learning_gain"))
        if learning_gain is not None:
            labels.append(row["student_id"])
            values.append(learning_gain)

    return _save_bar_chart(
        labels=labels,
        values=values,
        title="Learning Gain per Student",
        xlabel="Student ID",
        ylabel="Learning Gain",
        output_path=output_dir / "learning_gain_per_student.png",
    )


def plot_real_user_efficiency_metrics(
    rows: list[dict[str, Any]],
    output_dir: Path = REAL_USER_FIGURE_DIR,
) -> Path | None:
    response_time = _first_number(rows, "overall_average_response_time")
    attempts = _first_number(rows, "overall_average_attempts")
    if response_time is None or attempts is None:
        print("Warning: efficiency metrics are unavailable.")
        return None

    return _save_bar_chart(
        labels=["Response Time", "Attempts"],
        values=[response_time, attempts],
        title="Real-user Learning Efficiency Metrics",
        xlabel="Metric",
        ylabel="Average Value",
        output_path=output_dir / "real_user_efficiency_metrics.png",
    )


def plot_hint_effectiveness_by_level(
    rows: list[dict[str, Any]],
    output_dir: Path = REAL_USER_FIGURE_DIR,
) -> Path | None:
    correctness_values = [
        _first_number(rows, "level_1_correctness_rate"),
        _first_number(rows, "level_2_correctness_rate"),
        _first_number(rows, "level_3_correctness_rate"),
    ]

    if any(value is not None for value in correctness_values):
        values = [value if value is not None else 0.0 for value in correctness_values]
        ylabel = "Correctness Rate"
        title = "Hint Effectiveness by Hint Level"
    else:
        reward_values = [
            _first_number(rows, "level_1_average_reward"),
            _first_number(rows, "level_2_average_reward"),
            _first_number(rows, "level_3_average_reward"),
        ]
        if not any(value is not None for value in reward_values):
            print("Warning: hint effectiveness data is unavailable.")
            return None
        values = [value if value is not None else 0.0 for value in reward_values]
        ylabel = "Average Reward"
        title = "Average Reward by Hint Level"

    return _save_bar_chart(
        labels=["Level 1", "Level 2", "Level 3"],
        values=values,
        title=title,
        xlabel="Hint Level",
        ylabel=ylabel,
        output_path=output_dir / "hint_effectiveness_by_level.png",
    )


def plot_hint_usage_distribution_real_users(
    rows: list[dict[str, Any]],
    output_dir: Path = REAL_USER_FIGURE_DIR,
) -> Path | None:
    values = [
        _first_number(rows, "level_1_hint_count"),
        _first_number(rows, "level_2_hint_count"),
        _first_number(rows, "level_3_hint_count"),
    ]

    if not any(value is not None for value in values):
        print("Warning: hint usage data is unavailable.")
        return None

    return _save_bar_chart(
        labels=["Level 1", "Level 2", "Level 3"],
        values=[value if value is not None else 0.0 for value in values],
        title="Hint Usage Distribution for Real Users",
        xlabel="Hint Level",
        ylabel="Total Hint Count",
        output_path=output_dir / "hint_usage_distribution_real_users.png",
    )


def generate_real_user_figures(
    metrics_csv_path: str | Path = DEFAULT_METRICS_PATH,
    output_dir: str | Path = REAL_USER_FIGURE_DIR,
) -> list[Path]:
    """Generate all real-user evaluation figures from a metrics CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_metrics(metrics_csv_path)

    if not rows:
        print("Warning: no real-user metrics rows available. No figures were generated.")
        return []

    figure_paths = [
        plot_pre_post_score_comparison(rows, output_dir),
        plot_learning_gain_per_student(rows, output_dir),
        plot_real_user_efficiency_metrics(rows, output_dir),
        plot_hint_effectiveness_by_level(rows, output_dir),
        plot_hint_usage_distribution_real_users(rows, output_dir),
    ]

    return [path for path in figure_paths if path is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real-user experiment results.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=REAL_USER_FIGURE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved_paths = generate_real_user_figures(args.metrics, args.output_dir)
    for path in saved_paths:
        print(f"Saved real-user figure: {path}")
