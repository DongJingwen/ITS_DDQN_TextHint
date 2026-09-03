"""Generate corrected presentation-ready figures and source CSV files."""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "ppt_figures_clean"
CSV_DIR = OUTPUT_DIR / "source_data"
MOVING_AVERAGE_WINDOW = 20

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "figures" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.config import EPSILON_MIN, LAMBDA_ENTROPY


HINT_LABELS = {
    0: "Level 1: Strong Guidance",
    1: "Level 2: Partial Guidance",
    2: "Level 3: Minimal Scaffold",
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved source data: {path}")


def _row_count(path: Path) -> int:
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)
            return sum(1 for _ in reader)
    except OSError:
        return 0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _moving_average(values: list[float], window: int = MOVING_AVERAGE_WINDOW) -> list[float]:
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        subset = values[start:index + 1]
        smoothed.append(sum(subset) / len(subset))
    return smoothed


def _select_training_log() -> Path:
    default = LOG_DIR / "training_log.csv"
    if default.exists() and _row_count(default) > 0:
        return default
    candidates = [path for path in LOG_DIR.glob("training_log*.csv") if _row_count(path) > 0]
    if not candidates:
        raise FileNotFoundError("No non-empty training_log*.csv file found.")
    return max(candidates, key=_row_count)


def _select_interaction_log() -> Path | None:
    candidates = [
        LOG_DIR / "exports" / "interaction_logs.csv",
        LOG_DIR / "exports" / "demo" / "interaction_logs.csv",
    ]
    candidates.extend(LOG_DIR.glob("**/interaction_logs.csv"))
    usable = [path for path in candidates if path.exists() and _row_count(path) > 0]
    return max(usable, key=_row_count) if usable else None


def _save_line_with_ma(
    rows: list[dict[str, Any]],
    x_col: str,
    y_col: str,
    ma_col: str,
    title: str,
    y_label: str,
    output_name: str,
) -> Path:
    output = OUTPUT_DIR / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    x = [int(row[x_col]) for row in rows]
    y = [float(row[y_col]) for row in rows]
    ma = [float(row[ma_col]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=0.9, alpha=0.28, label="Raw")
    plt.plot(x, ma, linewidth=2.2, label=f"{MOVING_AVERAGE_WINDOW}-Episode Moving Average")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved figure: {output}")
    return output


def _save_bar(
    labels: list[str],
    values: list[float],
    title: str,
    y_label: str,
    output_name: str,
) -> Path:
    output = OUTPUT_DIR / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(0, max(values) * 1.2 if values else 1)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved figure: {output}")
    return output


def _save_grouped_percentage_bar(
    groups: list[str],
    rows: list[dict[str, Any]],
    title: str,
    x_label: str,
    output_name: str,
) -> Path:
    output = OUTPUT_DIR / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 0.24
    positions = list(range(len(groups)))
    by_group_action = {
        (row["group"], int(row["action"])): float(row["percentage"])
        for row in rows
    }

    plt.figure(figsize=(9, 5))
    for offset_index, action in enumerate((0, 1, 2)):
        values = [by_group_action.get((group, action), 0.0) for group in groups]
        offset = (offset_index - 1) * width
        plt.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=HINT_LABELS[action],
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Percentage within Group (%)")
    plt.xticks(positions, groups)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved figure: {output}")
    return output


def _training_rows() -> tuple[Path, list[dict[str, str]]]:
    path = _select_training_log()
    rows, _columns = _read_csv(path)
    return path, rows


def _series_from_training(rows: list[dict[str, str]], column: str) -> list[tuple[int, float]]:
    series = []
    for row in rows:
        episode = _to_float(row.get("episode"))
        value = _to_float(row.get(column))
        if episode is not None and value is not None:
            series.append((int(episode), value))
    return series


def generate_overall_hint_distribution(training_path: Path, rows: list[dict[str, str]]) -> Path:
    counts = Counter()
    for row in rows:
        for action in (0, 1, 2):
            counts[action] += _to_float(row.get(f"action_{action}_count")) or 0.0

    total = sum(counts.values())
    source_rows = []
    for action in (0, 1, 2):
        count = counts[action]
        percentage = 100.0 * count / total if total else 0.0
        source_rows.append(
            {
                "action": action,
                "hint_level": HINT_LABELS[action],
                "count": count,
                "percentage": percentage,
                "source_log": str(training_path),
            }
        )

    _write_csv(
        CSV_DIR / "overall_hint_level_distribution.csv",
        source_rows,
        ["action", "hint_level", "count", "percentage", "source_log"],
    )
    return _save_bar(
        [row["hint_level"] for row in source_rows],
        [row["percentage"] for row in source_rows],
        "Overall Hint Level Distribution",
        "Percentage of Recommendations (%)",
        "overall_hint_level_distribution.png",
    )


def _derive_entropy(epsilon: float) -> float:
    if LAMBDA_ENTROPY <= 0:
        return float("nan")
    return max(0.0, (epsilon - EPSILON_MIN) / LAMBDA_ENTROPY)


def generate_epsilon_entropy(training_path: Path, rows: list[dict[str, str]]) -> list[Path]:
    outputs = []
    epsilon_series = _series_from_training(rows, "average_epsilon")
    if not epsilon_series:
        print("Skipping epsilon/entropy figures: average_epsilon is missing.")
        return outputs

    entropy_logged = _series_from_training(rows, "average_entropy")
    entropy_by_episode = dict(entropy_logged)
    entropy_source = "logged_average_entropy" if entropy_logged else "derived_from_average_epsilon_current_formula"

    source_rows = []
    eps_values = []
    entropy_values = []
    for episode, epsilon in epsilon_series:
        entropy = entropy_by_episode.get(episode, _derive_entropy(epsilon))
        eps_values.append(epsilon)
        entropy_values.append(entropy)

    eps_ma = _moving_average(eps_values)
    entropy_ma = _moving_average(entropy_values)
    for index, (episode, epsilon) in enumerate(epsilon_series):
        source_rows.append(
            {
                "episode": episode,
                "average_epsilon": epsilon,
                "epsilon_moving_average_20": eps_ma[index],
                "entropy": entropy_values[index],
                "entropy_moving_average_20": entropy_ma[index],
                "entropy_source": entropy_source,
                "source_log": str(training_path),
            }
        )

    _write_csv(
        CSV_DIR / "dynamic_epsilon_entropy.csv",
        source_rows,
        [
            "episode",
            "average_epsilon",
            "epsilon_moving_average_20",
            "entropy",
            "entropy_moving_average_20",
            "entropy_source",
            "source_log",
        ],
    )
    outputs.append(_save_line_with_ma(
        source_rows,
        "episode",
        "average_epsilon",
        "epsilon_moving_average_20",
        "Dynamic Epsilon over Episodes",
        "Dynamic Epsilon",
        "dynamic_epsilon_curve_clean.png",
    ))
    outputs.append(_save_line_with_ma(
        source_rows,
        "episode",
        "entropy",
        "entropy_moving_average_20",
        "Entropy over Episodes",
        "Entropy of Q-value Distribution",
        "entropy_curve_clean.png",
    ))

    scatter_output = OUTPUT_DIR / "epsilon_vs_entropy_scatter.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(entropy_values, eps_values, alpha=0.6, s=18)
    plt.title("Dynamic Epsilon vs Entropy")
    plt.xlabel("Entropy of Q-value Distribution")
    plt.ylabel("Dynamic Epsilon")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(scatter_output, dpi=300)
    plt.close()
    print(f"Saved figure: {scatter_output}")
    outputs.append(scatter_output)
    return outputs


def _early_late_rows(series: list[tuple[int, float]], metric_name: str, source_log: Path) -> list[dict[str, Any]]:
    first = [value for _episode, value in series[:50]]
    last = [value for _episode, value in series[-50:]]
    return [
        {
            "period": "First 50 Episodes",
            metric_name: sum(first) / len(first) if first else 0.0,
            "n_episodes": len(first),
            "source_log": str(source_log),
        },
        {
            "period": "Last 50 Episodes",
            metric_name: sum(last) / len(last) if last else 0.0,
            "n_episodes": len(last),
            "source_log": str(source_log),
        },
    ]


def generate_efficiency(training_path: Path, rows: list[dict[str, str]]) -> list[Path]:
    outputs = []
    metrics = [
        ("average_attempts", "average_attempts", "Average Attempts over Episodes", "Average Attempts", "average_attempts_curve_clean.png", "average_attempts_early_vs_late.png"),
        ("average_response_time", "average_response_time", "Average Response Time over Episodes", "Average Response Time (seconds)", "average_response_time_curve_clean.png", "average_response_time_early_vs_late.png"),
    ]

    for column, metric_name, title, y_label, curve_name, bar_name in metrics:
        series = _series_from_training(rows, column)
        if not series:
            print(f"Skipping {title}: missing {column}.")
            continue
        episodes = [episode for episode, _value in series]
        values = [value for _episode, value in series]
        ma = _moving_average(values)
        source_rows = [
            {
                "episode": episodes[index],
                metric_name: values[index],
                f"{metric_name}_moving_average_20": ma[index],
                "source_log": str(training_path),
            }
            for index in range(len(episodes))
        ]
        _write_csv(
            CSV_DIR / f"{metric_name}_over_episodes.csv",
            source_rows,
            ["episode", metric_name, f"{metric_name}_moving_average_20", "source_log"],
        )
        outputs.append(_save_line_with_ma(
            source_rows,
            "episode",
            metric_name,
            f"{metric_name}_moving_average_20",
            title,
            y_label,
            curve_name,
        ))

        comparison_rows = _early_late_rows(series, metric_name, training_path)
        _write_csv(
            CSV_DIR / f"{metric_name}_early_vs_late.csv",
            comparison_rows,
            ["period", metric_name, "n_episodes", "source_log"],
        )
        outputs.append(_save_bar(
            [row["period"] for row in comparison_rows],
            [row[metric_name] for row in comparison_rows],
            f"{title}: First 50 vs Last 50 Episodes",
            y_label,
            bar_name,
        ))
    return outputs


def _interaction_rows() -> tuple[Path, list[dict[str, str]]] | None:
    path = _select_interaction_log()
    if path is None:
        return None
    rows, _columns = _read_csv(path)
    return path, rows


def _action_from_row(row: dict[str, str]) -> int | None:
    action = _to_float(row.get("action"))
    if action is not None and int(action) in HINT_LABELS:
        return int(action)
    hint = str(row.get("hint_level", "")).lower()
    if "level_1" in hint:
        return 0
    if "level_2" in hint:
        return 1
    if "level_3" in hint:
        return 2
    return None


def generate_mastery_figures(interaction: tuple[Path, list[dict[str, str]]] | None) -> list[Path]:
    outputs = []
    if interaction is None:
        print("Skipping mastery figures: no interaction log with rows found.")
        return outputs
    interaction_path, rows = interaction
    mastery_values = []
    usable_rows = []
    for row in rows:
        mastery = _to_float(row.get("mastery_level") or row.get("student_mastery") or row.get("current_level"))
        action = _action_from_row(row)
        if mastery is not None:
            mastery_values.append(mastery)
        if mastery is not None and action is not None:
            usable_rows.append((mastery, action))

    if not mastery_values:
        print("Skipping mastery figures: mastery_level/current_level is missing.")
        return outputs

    histogram_rows = [
        {"mastery_level": value, "source_log": str(interaction_path)}
        for value in mastery_values
    ]
    _write_csv(CSV_DIR / "mastery_values.csv", histogram_rows, ["mastery_level", "source_log"])

    histogram_output = OUTPUT_DIR / "mastery_histogram.png"
    plt.figure(figsize=(8, 5))
    plt.hist(mastery_values, bins=min(10, max(3, len(set(mastery_values)))))
    plt.title("Distribution of Student Mastery Values")
    plt.xlabel("Mastery Level")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(histogram_output, dpi=300)
    plt.close()
    print(f"Saved figure: {histogram_output}")
    outputs.append(histogram_output)

    fixed_groups = {"Low Mastery": [], "Medium Mastery": [], "High Mastery": []}
    for mastery, action in usable_rows:
        if mastery < 0.5:
            fixed_groups["Low Mastery"].append(action)
        elif mastery < 0.8:
            fixed_groups["Medium Mastery"].append(action)
        else:
            fixed_groups["High Mastery"].append(action)

    non_empty_groups = sum(1 for values in fixed_groups.values() if values)
    grouping_method = "fixed_thresholds"
    if non_empty_groups < 2 and len(usable_rows) >= 3:
        grouping_method = "tertile_rank_groups"
        sorted_rows = sorted(usable_rows, key=lambda item: item[0])
        fixed_groups = {"Low Mastery": [], "Medium Mastery": [], "High Mastery": []}
        for index, (_mastery, action) in enumerate(sorted_rows):
            tertile = int(index * 3 / len(sorted_rows))
            group = ("Low Mastery", "Medium Mastery", "High Mastery")[min(tertile, 2)]
            fixed_groups[group].append(action)

    source_rows = []
    for group, actions in fixed_groups.items():
        counts = Counter(actions)
        total = sum(counts.values())
        for action in (0, 1, 2):
            source_rows.append(
                {
                    "group": group,
                    "action": action,
                    "hint_level": HINT_LABELS[action],
                    "count": counts[action],
                    "percentage": 100.0 * counts[action] / total if total else 0.0,
                    "grouping_method": grouping_method,
                    "source_log": str(interaction_path),
                }
            )

    _write_csv(
        CSV_DIR / "hint_distribution_by_mastery.csv",
        source_rows,
        ["group", "action", "hint_level", "count", "percentage", "grouping_method", "source_log"],
    )
    outputs.append(_save_grouped_percentage_bar(
        ["Low Mastery", "Medium Mastery", "High Mastery"],
        source_rows,
        f"Hint Level Distribution by Student Mastery ({grouping_method})",
        "Student Mastery Group",
        "hint_distribution_by_mastery_clean.png",
    ))
    return outputs


def _difficulty_label(value: str) -> str | None:
    text = str(value).strip().lower()
    if text in {"low", "0", "0.0"}:
        return "Low Difficulty"
    if text in {"medium", "1", "1.0"}:
        return "Medium Difficulty"
    if text in {"high", "2", "2.0"}:
        return "High Difficulty"
    return None


def generate_difficulty_distribution(interaction: tuple[Path, list[dict[str, str]]] | None) -> list[Path]:
    outputs = []
    if interaction is None:
        print("Skipping difficulty distribution: no interaction log with rows found.")
        return outputs
    interaction_path, rows = interaction
    group_counts = {"Low Difficulty": Counter(), "Medium Difficulty": Counter(), "High Difficulty": Counter()}

    for row in rows:
        group = _difficulty_label(row.get("difficulty") or row.get("question_difficulty") or "")
        action = _action_from_row(row)
        if group is not None and action is not None:
            group_counts[group][action] += 1

    if not any(sum(counter.values()) for counter in group_counts.values()):
        print("Skipping difficulty distribution: difficulty/action columns missing or empty.")
        return outputs

    source_rows = []
    for group, counts in group_counts.items():
        total = sum(counts.values())
        for action in (0, 1, 2):
            source_rows.append(
                {
                    "group": group,
                    "action": action,
                    "hint_level": HINT_LABELS[action],
                    "count": counts[action],
                    "percentage": 100.0 * counts[action] / total if total else 0.0,
                    "source_log": str(interaction_path),
                }
            )
    _write_csv(
        CSV_DIR / "hint_distribution_by_difficulty.csv",
        source_rows,
        ["group", "action", "hint_level", "count", "percentage", "source_log"],
    )
    outputs.append(_save_grouped_percentage_bar(
        ["Low Difficulty", "Medium Difficulty", "High Difficulty"],
        source_rows,
        "Hint Level Distribution by Question Difficulty",
        "Question Difficulty",
        "hint_distribution_by_difficulty_clean.png",
    ))
    return outputs


def generate_reward_loss(training_path: Path, rows: list[dict[str, str]]) -> list[Path]:
    outputs = []
    for column, metric_name, title, y_label, output_name in (
        ("total_reward", "total_reward", "Training Reward over Episodes", "Episode Reward", "reward_curve_clean.png"),
        ("average_loss", "average_loss", "Training Loss over Episodes", "Training Loss", "loss_curve_clean.png"),
    ):
        series = _series_from_training(rows, column)
        if not series:
            print(f"Skipping {title}: missing {column}.")
            continue
        episodes = [episode for episode, _value in series]
        values = [value for _episode, value in series]
        ma = _moving_average(values)
        source_rows = [
            {
                "episode": episodes[index],
                metric_name: values[index],
                f"{metric_name}_moving_average_20": ma[index],
                "source_log": str(training_path),
            }
            for index in range(len(episodes))
        ]
        _write_csv(
            CSV_DIR / f"{metric_name}_over_episodes.csv",
            source_rows,
            ["episode", metric_name, f"{metric_name}_moving_average_20", "source_log"],
        )
        outputs.append(_save_line_with_ma(
            source_rows,
            "episode",
            metric_name,
            f"{metric_name}_moving_average_20",
            title,
            y_label,
            output_name,
        ))
    return outputs


def generate_clean_ppt_figures() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    training_path, training_rows = _training_rows()
    interaction = _interaction_rows()

    print(f"Selected training log: {training_path}")
    if interaction is None:
        print("Selected interaction log: Not available")
    else:
        print(f"Selected interaction log: {interaction[0]}")

    outputs: list[Path] = []
    outputs.append(generate_overall_hint_distribution(training_path, training_rows))
    outputs.extend(generate_epsilon_entropy(training_path, training_rows))
    outputs.extend(generate_efficiency(training_path, training_rows))
    outputs.extend(generate_mastery_figures(interaction))
    outputs.extend(generate_difficulty_distribution(interaction))
    outputs.extend(generate_reward_loss(training_path, training_rows))

    print("\nRemoved figure: average_hint_usage_per_question.png")
    print("Reason: training interactions contain exactly one recommended hint action per step; no meaningful hint request/trigger metric is logged.")
    print("Future logging note: record whether a hint was requested/triggered separately from the recommended action.")

    return outputs


def main() -> int:
    try:
        outputs = generate_clean_ppt_figures()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nGenerated clean PPT figures:")
    for output in outputs:
        print(f"- {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
