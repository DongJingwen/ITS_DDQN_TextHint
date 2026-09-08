"""Ablation study comparing dynamic, fixed, and linear-decay epsilon strategies."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent.double_dqn import DoubleDQNAgent
from agent.replay_buffer import ReplayBuffer
from config.config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    EPSILON_FIXED,
    MAX_STEPS_PER_EPISODE,
)
from env.environment import TutoringEnvironment
from training.run_episode import run_episode


LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "ablation"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "ablation"
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
Path(FIGURE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ABLATION_SETTINGS = [
    ("dynamic_entropy", "DDQN + Dynamic Entropy Epsilon"),
    ("fixed", "DDQN + Fixed Epsilon"),
    ("linear_decay", "DDQN + Linear Decay Epsilon"),
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_episode_log(log_path: Path, episode_logs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode",
        "epsilon_mode",
        "total_reward",
        "average_loss",
        "average_epsilon",
        "average_entropy",
        "average_attempts",
        "average_response_time",
        "action_0_count",
        "action_1_count",
        "action_2_count",
        "steps",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_logs)


def _read_log(log_path: Path) -> list[dict[str, Any]]:
    with log_path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _float_column(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def _sum_column(rows: list[dict[str, Any]], column: str) -> float:
    return sum(float(row[column]) for row in rows)


def run_ablation_setting(
    epsilon_mode: str,
    num_episodes: int,
    max_steps_per_episode: int,
    batch_size: int,
    buffer_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Train one DDQN variant and return per-episode logs."""
    _set_seed(seed)
    environment = TutoringEnvironment(max_steps=max_steps_per_episode, seed=seed)
    agent = DoubleDQNAgent(
        epsilon_mode=epsilon_mode,
        epsilon_fixed=EPSILON_FIXED,
        device="cpu",
    )
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    logs: list[dict[str, Any]] = []

    for episode in range(1, num_episodes + 1):
        agent.set_training_progress(episode, num_episodes)
        summary = run_episode(
            environment=environment,
            agent=agent,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            train=True,
        )
        summary["episode"] = episode
        summary["epsilon_mode"] = epsilon_mode
        logs.append(summary)

    return logs


def _plot_line_comparison(
    logs_by_mode: dict[str, list[dict[str, Any]]],
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    plt.figure(figsize=(9, 5), dpi=300)
    for epsilon_mode, label in ABLATION_SETTINGS:
        rows = logs_by_mode[epsilon_mode]
        episodes = _float_column(rows, "episode")
        values = _float_column(rows, column)
        plt.plot(episodes, values, linewidth=2, label=label)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _plot_efficiency_comparison(
    logs_by_mode: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> Path:
    labels = [label for _, label in ABLATION_SETTINGS]
    attempts = [
        np.mean(_float_column(logs_by_mode[mode], "average_attempts"))
        for mode, _ in ABLATION_SETTINGS
    ]
    response_times = [
        np.mean(_float_column(logs_by_mode[mode], "average_response_time"))
        for mode, _ in ABLATION_SETTINGS
    ]

    x_positions = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(x_positions - width / 2, attempts, width, label="Average Attempts")
    plt.bar(x_positions + width / 2, response_times, width, label="Average Response Time")
    plt.title("Ablation Efficiency Comparison")
    plt.xlabel("Exploration Strategy")
    plt.ylabel("Average Value")
    plt.xticks(x_positions, labels, rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _plot_action_distribution(
    logs_by_mode: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> Path:
    labels = [label for _, label in ABLATION_SETTINGS]
    action_0 = [_sum_column(logs_by_mode[mode], "action_0_count") for mode, _ in ABLATION_SETTINGS]
    action_1 = [_sum_column(logs_by_mode[mode], "action_1_count") for mode, _ in ABLATION_SETTINGS]
    action_2 = [_sum_column(logs_by_mode[mode], "action_2_count") for mode, _ in ABLATION_SETTINGS]

    x_positions = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(x_positions - width, action_0, width, label="Action 0: Strong")
    plt.bar(x_positions, action_1, width, label="Action 1: Medium")
    plt.bar(x_positions + width, action_2, width, label="Action 2: Weak")
    plt.title("Ablation Hint Action Distribution")
    plt.xlabel("Exploration Strategy")
    plt.ylabel("Total Action Count")
    plt.xticks(x_positions, labels, rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_ablation_plots(logs_by_mode: dict[str, list[dict[str, Any]]]) -> list[Path]:
    """Generate required ablation comparison figures."""
    return [
        _plot_line_comparison(
            logs_by_mode,
            column="total_reward",
            title="Ablation Reward Comparison",
            ylabel="Total Reward",
            output_path=FIGURE_DIR / "ablation_reward_comparison.png",
        ),
        _plot_line_comparison(
            logs_by_mode,
            column="average_epsilon",
            title="Ablation Epsilon Comparison",
            ylabel="Average Epsilon",
            output_path=FIGURE_DIR / "ablation_epsilon_comparison.png",
        ),
        _plot_efficiency_comparison(
            logs_by_mode,
            output_path=FIGURE_DIR / "ablation_efficiency_comparison.png",
        ),
        _plot_action_distribution(
            logs_by_mode,
            output_path=FIGURE_DIR / "ablation_action_distribution.png",
        ),
    ]


def run_ablation_study(
    num_episodes: int = 50,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    batch_size: int = BATCH_SIZE,
    buffer_size: int = BUFFER_SIZE,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all epsilon ablation settings and save logs plus figures."""
    logs_by_mode = {}
    log_paths = {}

    for epsilon_mode, _ in ABLATION_SETTINGS:
        logs = run_ablation_setting(
            epsilon_mode=epsilon_mode,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            batch_size=batch_size,
            buffer_size=buffer_size,
            seed=seed,
        )
        logs_by_mode[epsilon_mode] = logs
        log_path = LOG_DIR / f"{epsilon_mode}.csv"
        _write_episode_log(log_path, logs)
        log_paths[epsilon_mode] = log_path
        print(f"Saved ablation log for {epsilon_mode}: {log_path}")

    figure_paths = generate_ablation_plots(logs_by_mode)
    for path in figure_paths:
        print(f"Saved ablation figure: {path}")

    return {
        "logs_by_mode": logs_by_mode,
        "log_paths": log_paths,
        "figure_paths": figure_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run epsilon strategy ablation study.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_study(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )
