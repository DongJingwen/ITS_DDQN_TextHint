"""Generate thesis-ready meeting analysis figures from existing logs."""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "meeting_analysis"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "figures" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


COLUMN_CANDIDATES = {
    "episode": ("episode", "Episode"),
    "step": ("step", "steps", "step_count"),
    "action": ("action", "selected_action"),
    "hint_level": ("hint_level", "hint"),
    "epsilon": ("epsilon", "average_epsilon", "avg_epsilon"),
    "entropy": ("entropy", "average_entropy", "avg_entropy"),
    "response_time": ("response_time", "average_response_time", "avg_response_time"),
    "attempts": ("attempts", "average_attempts", "avg_attempts"),
    "hint_usage": ("hint_count", "hints_used", "hint_usage", "number_of_hints"),
    "mastery": ("student_mastery", "mastery_level", "current_level"),
    "difficulty": ("question_difficulty", "difficulty"),
    "question_id": ("question_id",),
}

HINT_LABELS = {
    0: "Level 1 Strong Hint",
    1: "Level 2 Medium Hint",
    2: "Level 3 Weak Hint",
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def _row_count(path: Path) -> int:
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)
            return sum(1 for _ in reader)
    except OSError:
        return 0


def _find_column(columns: list[str], key: str) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in COLUMN_CANDIDATES[key]:
        if candidate in columns:
            return candidate
        match = lower_to_original.get(candidate.lower())
        if match:
            return match
    return None


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


def _to_action(row: dict[str, str], action_col: str | None, hint_col: str | None) -> int | None:
    if action_col is not None:
        value = _to_float(row.get(action_col))
        if value is not None and int(value) in HINT_LABELS:
            return int(value)

    if hint_col is not None:
        text = str(row.get(hint_col, "")).lower()
        if "level_1" in text or "level 1" in text or text == "1":
            return 0
        if "level_2" in text or "level 2" in text or text == "2":
            return 1
        if "level_3" in text or "level 3" in text or text == "3":
            return 2
    return None


def _difficulty_label(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"low", "0", "0.0"}:
        return "Low difficulty"
    if text in {"medium", "1", "1.0"}:
        return "Medium difficulty"
    if text in {"high", "2", "2.0"}:
        return "High difficulty"
    return None


def _mastery_group(value: Any) -> str | None:
    mastery = _to_float(value)
    if mastery is None:
        return None
    if mastery < 0.50:
        return "Low mastery"
    if mastery < 0.80:
        return "Medium mastery"
    return "High mastery"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _select_longest_training_log() -> Path | None:
    default = LOG_DIR / "training_log.csv"
    if default.exists():
        return default
    candidates = sorted(LOG_DIR.glob("training_log*.csv"))
    return max(candidates, key=_row_count) if candidates else None


def _select_interaction_log() -> Path | None:
    candidates = [
        LOG_DIR / "exports" / "interaction_logs.csv",
        LOG_DIR / "exports" / "demo" / "interaction_logs.csv",
    ]
    candidates.extend(sorted(LOG_DIR.glob("**/interaction_logs.csv")))
    usable = [path for path in candidates if path.exists() and _row_count(path) > 0]
    return max(usable, key=_row_count) if usable else None


def _save_line_plot(
    x_values: list[int],
    y_values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_name: str,
) -> Path:
    output_path = FIGURE_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, linewidth=1.8, marker="o" if len(x_values) == 1 else None)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")
    return output_path


def _save_bar_plot(
    labels: list[str],
    values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_name: str,
) -> Path:
    output_path = FIGURE_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")
    return output_path


def _save_grouped_bar_plot(
    group_labels: list[str],
    counts_by_group: dict[str, Counter],
    title: str,
    x_label: str,
    y_label: str,
    output_name: str,
) -> Path:
    output_path = FIGURE_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(group_labels)))
    width = 0.24

    plt.figure(figsize=(9, 5))
    for offset_index, action in enumerate((0, 1, 2)):
        offset = (offset_index - 1) * width
        values = [counts_by_group[group][action] for group in group_labels]
        plt.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=HINT_LABELS[action],
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_positions, group_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")
    return output_path


def _aggregate_by_episode(
    rows: list[dict[str, str]],
    episode_col: str,
    value_col: str,
) -> tuple[list[int], list[float]]:
    values_by_episode: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        episode_value = _to_float(row.get(episode_col))
        metric_value = _to_float(row.get(value_col))
        if episode_value is None or metric_value is None:
            continue
        values_by_episode[int(episode_value)].append(metric_value)

    episodes = sorted(values_by_episode)
    values = [_mean(values_by_episode[episode]) for episode in episodes]
    return episodes, values


def plot_dynamic_epsilon(training: tuple[Path, list[dict[str, str]], list[str]] | None, report: list[str]) -> Path | None:
    if training is None:
        report.append("SKIPPED dynamic epsilon: no training log found.")
        return None
    path, rows, columns = training
    episode_col = _find_column(columns, "episode")
    epsilon_col = _find_column(columns, "epsilon")
    if episode_col is None or epsilon_col is None:
        report.append(f"SKIPPED dynamic epsilon: missing episode/epsilon columns in {path}.")
        return None
    episodes, epsilons = _aggregate_by_episode(rows, episode_col, epsilon_col)
    if not episodes:
        report.append(f"SKIPPED dynamic epsilon: no numeric epsilon values in {path}.")
        return None
    report.append(f"Dynamic epsilon: {path} columns episode={episode_col}, epsilon={epsilon_col}")
    return _save_line_plot(
        episodes,
        epsilons,
        "Dynamic Epsilon over Episodes",
        "Episode",
        "Dynamic epsilon",
        "dynamic_epsilon_curve.png",
    )


def plot_entropy(training: tuple[Path, list[dict[str, str]], list[str]] | None, report: list[str]) -> Path | None:
    if training is None:
        report.append("SKIPPED entropy: no training log found.")
        return None
    path, rows, columns = training
    episode_col = _find_column(columns, "episode")
    entropy_col = _find_column(columns, "entropy")
    if episode_col is None or entropy_col is None:
        report.append(f"SKIPPED entropy: no entropy or average_entropy column in {path}.")
        return None
    episodes, entropies = _aggregate_by_episode(rows, episode_col, entropy_col)
    if not episodes:
        report.append(f"SKIPPED entropy: no numeric entropy values in {path}.")
        return None
    report.append(f"Entropy: {path} columns episode={episode_col}, entropy={entropy_col}")
    return _save_line_plot(
        episodes,
        entropies,
        "Entropy of Q-value Distribution over Episodes",
        "Episode",
        "Entropy of Q-value Distribution",
        "entropy_curve.png",
    )


def plot_overall_hint_distribution(
    training: tuple[Path, list[dict[str, str]], list[str]] | None,
    interaction: tuple[Path, list[dict[str, str]], list[str]] | None,
    report: list[str],
) -> Path | None:
    counts = Counter()
    source_text = ""

    if training is not None:
        path, rows, columns = training
        count_cols = {action: f"action_{action}_count" for action in (0, 1, 2)}
        if all(column in columns for column in count_cols.values()):
            for row in rows:
                for action, column in count_cols.items():
                    value = _to_float(row.get(column))
                    if value is not None:
                        counts[action] += value
            source_text = f"{path} columns {count_cols}"

    if not counts and interaction is not None:
        path, rows, columns = interaction
        action_col = _find_column(columns, "action")
        hint_col = _find_column(columns, "hint_level")
        for row in rows:
            action = _to_action(row, action_col, hint_col)
            if action is not None:
                counts[action] += 1
        source_text = f"{path} columns action={action_col}, hint_level={hint_col}"

    if not counts:
        report.append("SKIPPED overall hint distribution: no action or hint columns found.")
        return None

    report.append(f"Overall hint distribution: {source_text}")
    return _save_bar_plot(
        [HINT_LABELS[action] for action in (0, 1, 2)],
        [counts[action] for action in (0, 1, 2)],
        "Overall Hint Level Distribution",
        "Hint Level",
        "Number of Recommendations",
        "overall_hint_level_distribution.png",
    )


def plot_hint_distribution_by_mastery(interaction: tuple[Path, list[dict[str, str]], list[str]] | None, report: list[str]) -> Path | None:
    if interaction is None:
        report.append("SKIPPED mastery hint distribution: no interaction log found.")
        return None
    path, rows, columns = interaction
    action_col = _find_column(columns, "action")
    hint_col = _find_column(columns, "hint_level")
    mastery_col = _find_column(columns, "mastery")
    if mastery_col is None or (action_col is None and hint_col is None):
        report.append(f"SKIPPED mastery hint distribution: missing mastery/action columns in {path}.")
        return None

    group_order = ["Low mastery", "Medium mastery", "High mastery"]
    counts_by_group = {group: Counter() for group in group_order}
    for row in rows:
        group = _mastery_group(row.get(mastery_col))
        action = _to_action(row, action_col, hint_col)
        if group is not None and action is not None:
            counts_by_group[group][action] += 1

    if sum(sum(counter.values()) for counter in counts_by_group.values()) == 0:
        report.append(f"SKIPPED mastery hint distribution: no usable rows in {path}.")
        return None

    report.append(f"Hint by mastery: {path} columns mastery={mastery_col}, action={action_col}, hint_level={hint_col}")
    return _save_grouped_bar_plot(
        group_order,
        counts_by_group,
        "Hint Level Distribution by Student Mastery Level",
        "Student Mastery Group",
        "Number of Recommendations",
        "hint_distribution_by_mastery.png",
    )


def plot_hint_distribution_by_difficulty(interaction: tuple[Path, list[dict[str, str]], list[str]] | None, report: list[str]) -> Path | None:
    if interaction is None:
        report.append("SKIPPED difficulty hint distribution: no interaction log found.")
        return None
    path, rows, columns = interaction
    action_col = _find_column(columns, "action")
    hint_col = _find_column(columns, "hint_level")
    difficulty_col = _find_column(columns, "difficulty")
    if difficulty_col is None or (action_col is None and hint_col is None):
        report.append(f"SKIPPED difficulty hint distribution: missing difficulty/action columns in {path}.")
        return None

    group_order = ["Low difficulty", "Medium difficulty", "High difficulty"]
    counts_by_group = {group: Counter() for group in group_order}
    for row in rows:
        group = _difficulty_label(row.get(difficulty_col))
        action = _to_action(row, action_col, hint_col)
        if group is not None and action is not None:
            counts_by_group[group][action] += 1

    if sum(sum(counter.values()) for counter in counts_by_group.values()) == 0:
        report.append(f"SKIPPED difficulty hint distribution: no usable rows in {path}.")
        return None

    report.append(f"Hint by difficulty: {path} columns difficulty={difficulty_col}, action={action_col}, hint_level={hint_col}")
    return _save_grouped_bar_plot(
        group_order,
        counts_by_group,
        "Hint Level Distribution by Question Difficulty",
        "Question Difficulty",
        "Number of Recommendations",
        "hint_distribution_by_difficulty.png",
    )


def plot_efficiency_line(
    training: tuple[Path, list[dict[str, str]], list[str]] | None,
    metric_key: str,
    title: str,
    y_label: str,
    output_name: str,
    report: list[str],
) -> Path | None:
    if training is None:
        report.append(f"SKIPPED {title}: no training log found.")
        return None
    path, rows, columns = training
    episode_col = _find_column(columns, "episode")
    metric_col = _find_column(columns, metric_key)
    if episode_col is None or metric_col is None:
        report.append(f"SKIPPED {title}: missing episode/{metric_key} columns in {path}.")
        return None
    episodes, values = _aggregate_by_episode(rows, episode_col, metric_col)
    if not episodes:
        report.append(f"SKIPPED {title}: no numeric values in {path}.")
        return None
    report.append(f"{title}: {path} columns episode={episode_col}, {metric_key}={metric_col}")
    return _save_line_plot(episodes, values, title, "Episode", y_label, output_name)


def plot_average_hint_usage(
    training: tuple[Path, list[dict[str, str]], list[str]] | None,
    interaction: tuple[Path, list[dict[str, str]], list[str]] | None,
    report: list[str],
) -> Path | None:
    if training is not None:
        path, rows, columns = training
        episode_col = _find_column(columns, "episode")
        step_col = _find_column(columns, "step")
        count_cols = {action: f"action_{action}_count" for action in (0, 1, 2)}
        if episode_col is not None and step_col is not None and all(column in columns for column in count_cols.values()):
            episodes = []
            values = []
            for row in rows:
                episode = _to_float(row.get(episode_col))
                steps = _to_float(row.get(step_col))
                if episode is None or steps is None or steps <= 0:
                    continue
                hint_total = sum(_to_float(row.get(column)) or 0.0 for column in count_cols.values())
                episodes.append(int(episode))
                values.append(hint_total / steps)
            if len(episodes) > 1:
                report.append(f"Average hint usage: {path} approximated from action counts / steps because question_id is unavailable.")
                return _save_line_plot(
                    episodes,
                    values,
                    "Average Hint Usage per Question",
                    "Episode",
                    "Average Number of Hints Used per Question",
                    "average_hint_usage_per_question.png",
                )

    if interaction is not None:
        path, rows, columns = interaction
        episode_col = _find_column(columns, "episode")
        question_col = _find_column(columns, "question_id")
        action_col = _find_column(columns, "action")
        hint_col = _find_column(columns, "hint_level")
        usage_col = _find_column(columns, "hint_usage")

        if episode_col is not None and usage_col is not None:
            episodes, values = _aggregate_by_episode(rows, episode_col, usage_col)
            if episodes:
                report.append(f"Average hint usage: {path} columns episode={episode_col}, hint_usage={usage_col}")
                return _save_line_plot(
                    episodes,
                    values,
                    "Average Hint Usage per Question",
                    "Episode",
                    "Average Number of Hints Used per Question",
                    "average_hint_usage_per_question.png",
                )

        if episode_col is not None and question_col is not None and (action_col is not None or hint_col is not None):
            hints_by_episode_question: dict[tuple[int, str], int] = defaultdict(int)
            for row in rows:
                episode_value = _to_float(row.get(episode_col))
                question_id = row.get(question_col)
                action = _to_action(row, action_col, hint_col)
                if episode_value is None or not question_id or action is None:
                    continue
                hints_by_episode_question[(int(episode_value), question_id)] += 1

            values_by_episode: dict[int, list[float]] = defaultdict(list)
            for (episode, _question_id), count in hints_by_episode_question.items():
                values_by_episode[episode].append(float(count))
            episodes = sorted(values_by_episode)
            values = [_mean(values_by_episode[episode]) for episode in episodes]
            if episodes:
                report.append(f"Average hint usage: {path} columns episode={episode_col}, question_id={question_col}, action={action_col}, hint_level={hint_col}")
                return _save_line_plot(
                    episodes,
                    values,
                    "Average Hint Usage per Question",
                    "Episode",
                    "Average Number of Hints Used per Question",
                    "average_hint_usage_per_question.png",
                )

    if training is not None:
        path, rows, columns = training
        episode_col = _find_column(columns, "episode")
        step_col = _find_column(columns, "step")
        count_cols = {action: f"action_{action}_count" for action in (0, 1, 2)}
        if episode_col is not None and step_col is not None and all(column in columns for column in count_cols.values()):
            episodes = []
            values = []
            for row in rows:
                episode = _to_float(row.get(episode_col))
                steps = _to_float(row.get(step_col))
                if episode is None or steps is None or steps <= 0:
                    continue
                hint_total = sum(_to_float(row.get(column)) or 0.0 for column in count_cols.values())
                episodes.append(int(episode))
                values.append(hint_total / steps)
            if episodes:
                report.append(f"Average hint usage: {path} approximated from action counts / steps because question_id is unavailable.")
                return _save_line_plot(
                    episodes,
                    values,
                    "Average Hint Usage per Question",
                    "Episode",
                    "Average Number of Hints Used per Question",
                    "average_hint_usage_per_question.png",
                )

    report.append("SKIPPED average hint usage: no hint usage, action/question, or action count data found.")
    return None


def load_sources() -> tuple[
    tuple[Path, list[dict[str, str]], list[str]] | None,
    tuple[Path, list[dict[str, str]], list[str]] | None,
]:
    training_path = _select_longest_training_log()
    interaction_path = _select_interaction_log()

    training = None
    interaction = None
    if training_path is not None:
        training_rows, training_columns = _read_csv(training_path)
        training = (training_path, training_rows, training_columns)
    if interaction_path is not None:
        interaction_rows, interaction_columns = _read_csv(interaction_path)
        interaction = (interaction_path, interaction_rows, interaction_columns)
    return training, interaction


def generate_meeting_analysis_figures() -> tuple[list[Path], list[str]]:
    training, interaction = load_sources()
    report: list[str] = []
    saved: list[Path] = []

    if training is not None:
        report.append(f"Selected training log: {training[0]}")
    else:
        report.append("Selected training log: Not available")
    if interaction is not None:
        report.append(f"Selected interaction log: {interaction[0]}")
    else:
        report.append("Selected interaction log: Not available")

    for plotter in (
        lambda: plot_dynamic_epsilon(training, report),
        lambda: plot_entropy(training, report),
        lambda: plot_overall_hint_distribution(training, interaction, report),
        lambda: plot_hint_distribution_by_mastery(interaction, report),
        lambda: plot_hint_distribution_by_difficulty(interaction, report),
        lambda: plot_efficiency_line(
            training,
            "response_time",
            "Average Response Time over Episodes",
            "Average Response Time",
            "average_response_time_over_episodes.png",
            report,
        ),
        lambda: plot_efficiency_line(
            training,
            "attempts",
            "Average Attempts over Episodes",
            "Average Number of Attempts",
            "average_attempts_over_episodes.png",
            report,
        ),
        lambda: plot_average_hint_usage(training, interaction, report),
    ):
        output = plotter()
        if output is not None:
            saved.append(output)

    return saved, report


def main() -> int:
    try:
        saved, report = generate_meeting_analysis_figures()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nAnalysis figure report:")
    for line in report:
        print(f"- {line}")

    if not saved:
        print("\nNo figures were generated.")
    else:
        print("\nGenerated figures:")
        for path in saved:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
