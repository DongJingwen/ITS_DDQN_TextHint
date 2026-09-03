"""Generate thesis-ready RL training performance curves.

This script reads an existing training log and creates:
- cumulative reward curve
- average reward per episode curve
- loss curve
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "training_log.csv"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


EPISODE_COLUMNS = ("episode", "Episode")
TOTAL_REWARD_COLUMNS = ("total_reward", "episode_reward", "reward", "Total Reward")
AVERAGE_REWARD_COLUMNS = ("average_reward", "avg_reward", "mean_reward")
LOSS_COLUMNS = ("average_loss", "avg_loss", "loss", "training_loss")
STEPS_COLUMNS = ("steps", "step_count", "num_steps")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def _read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Training log has no data rows: {csv_path}")

    return rows, fieldnames


def _count_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)
            return sum(1 for _ in reader)
    except OSError:
        return 0


def choose_training_log(log_path: str | Path | None = None) -> Path:
    """Choose the requested log, default log, or the longest training_log*.csv."""
    if log_path is not None:
        return Path(log_path)

    if DEFAULT_LOG_PATH.exists():
        return DEFAULT_LOG_PATH

    candidates = sorted(LOG_DIR.glob("training_log*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No training log found. Expected {DEFAULT_LOG_PATH} or training_log*.csv under {LOG_DIR}."
        )

    return max(candidates, key=_count_rows)


def aggregate_training_log(csv_path: Path) -> dict[str, Any]:
    """Load and aggregate training log columns by episode."""
    rows, fieldnames = _read_csv_rows(csv_path)

    episode_col = _find_column(fieldnames, EPISODE_COLUMNS)
    total_reward_col = _find_column(fieldnames, TOTAL_REWARD_COLUMNS)
    average_reward_col = _find_column(fieldnames, AVERAGE_REWARD_COLUMNS)
    loss_col = _find_column(fieldnames, LOSS_COLUMNS)
    steps_col = _find_column(fieldnames, STEPS_COLUMNS)

    missing = []
    if episode_col is None:
        missing.append("episode")
    if total_reward_col is None:
        missing.append("total_reward or reward")
    if loss_col is None:
        missing.append("average_loss or loss")
    if missing:
        raise ValueError(
            "Training log is missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {fieldnames}"
        )

    total_reward_by_episode: dict[int, float] = defaultdict(float)
    average_reward_values: dict[int, list[float]] = defaultdict(list)
    loss_values: dict[int, list[float]] = defaultdict(list)
    steps_by_episode: dict[int, float] = defaultdict(float)

    for row in rows:
        episode_value = _to_float(row.get(episode_col))
        if episode_value is None:
            continue
        episode = int(episode_value)

        total_reward = _to_float(row.get(total_reward_col))
        if total_reward is not None:
            total_reward_by_episode[episode] += total_reward

        if average_reward_col is not None:
            average_reward = _to_float(row.get(average_reward_col))
            if average_reward is not None:
                average_reward_values[episode].append(average_reward)

        loss = _to_float(row.get(loss_col))
        if loss is not None:
            loss_values[episode].append(loss)

        if steps_col is not None:
            steps = _to_float(row.get(steps_col))
            if steps is not None:
                steps_by_episode[episode] += steps

    episodes = sorted(total_reward_by_episode)
    if not episodes:
        raise ValueError("No valid episode/reward rows were found in the training log.")

    total_rewards = [total_reward_by_episode[episode] for episode in episodes]
    cumulative_rewards = []
    running_total = 0.0
    for reward in total_rewards:
        running_total += reward
        cumulative_rewards.append(running_total)

    average_reward_label = "Average Reward per Episode"
    if average_reward_col is not None and average_reward_values:
        average_rewards = [
            sum(average_reward_values[episode]) / len(average_reward_values[episode])
            if average_reward_values[episode]
            else 0.0
            for episode in episodes
        ]
    elif steps_col is not None and steps_by_episode:
        average_rewards = [
            total_reward_by_episode[episode] / steps_by_episode[episode]
            if steps_by_episode[episode] > 0
            else 0.0
            for episode in episodes
        ]
    else:
        average_rewards = total_rewards
        average_reward_label = "Episode Reward (total_reward)"

    loss_episodes = [episode for episode in episodes if loss_values[episode]]
    losses = [
        sum(loss_values[episode]) / len(loss_values[episode])
        for episode in loss_episodes
    ]

    if not losses:
        raise ValueError(
            f"No numeric loss values found in column '{loss_col}'."
        )

    return {
        "csv_path": csv_path,
        "episodes": episodes,
        "total_rewards": total_rewards,
        "cumulative_rewards": cumulative_rewards,
        "average_rewards": average_rewards,
        "average_reward_label": average_reward_label,
        "loss_episodes": loss_episodes,
        "losses": losses,
        "columns_used": {
            "episode": episode_col,
            "total_reward": total_reward_col,
            "average_reward": average_reward_col,
            "loss": loss_col,
            "steps": steps_col,
        },
    }


def _save_line_plot(
    x_values: list[int],
    y_values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, linewidth=1.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def generate_rl_training_curves(log_path: str | Path | None = None) -> dict[str, Path]:
    selected_log = choose_training_log(log_path)
    data = aggregate_training_log(selected_log)

    print(f"Using training log: {data['csv_path']}")
    print(f"Columns used: {data['columns_used']}")

    cumulative_reward_path = FIGURE_DIR / "cumulative_reward_curve.png"
    average_reward_path = FIGURE_DIR / "average_reward_per_episode_curve.png"
    loss_path = FIGURE_DIR / "loss_curve.png"

    _save_line_plot(
        x_values=data["episodes"],
        y_values=data["cumulative_rewards"],
        title="Cumulative Reward Curve",
        x_label="Episode",
        y_label="Cumulative Reward",
        output_path=cumulative_reward_path,
    )
    _save_line_plot(
        x_values=data["episodes"],
        y_values=data["average_rewards"],
        title="Average Reward per Episode Curve",
        x_label="Episode",
        y_label=data["average_reward_label"],
        output_path=average_reward_path,
    )
    _save_line_plot(
        x_values=data["loss_episodes"],
        y_values=data["losses"],
        title="Training Loss Curve",
        x_label="Episode",
        y_label="Training Loss",
        output_path=loss_path,
    )

    return {
        "cumulative_reward": cumulative_reward_path,
        "average_reward": average_reward_path,
        "loss": loss_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RL training performance curves from an existing CSV log."
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional path to a training log CSV. Defaults to outputs/logs/training_log.csv or the longest training_log*.csv.",
    )
    args = parser.parse_args()

    try:
        generate_rl_training_curves(args.log)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
