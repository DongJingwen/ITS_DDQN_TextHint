"""Training entry point for the DDQN text-hint tutoring prototype."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent.double_dqn import DoubleDQNAgent
from agent.replay_buffer import ReplayBuffer
from config import config as config_module
from config.config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    MAX_STEPS_PER_EPISODE,
    NUM_EPISODES,
)
from env.environment import TutoringEnvironment
from training.run_episode import run_episode
from utils.reproducibility import save_experiment_config, set_global_seed


LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_logs(log_path: Path, episode_logs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode",
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


def train(
    num_episodes: int = NUM_EPISODES,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    batch_size: int = BATCH_SIZE,
    buffer_size: int = BUFFER_SIZE,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Train the Double DQN agent and save CSV logs plus model weights."""
    set_global_seed(seed)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    environment = TutoringEnvironment(max_steps=max_steps_per_episode, seed=seed)
    agent = DoubleDQNAgent()
    replay_buffer = ReplayBuffer(capacity=buffer_size)

    episode_logs: list[dict[str, Any]] = []

    for episode in range(1, num_episodes + 1):
        episode_summary = run_episode(
            environment=environment,
            agent=agent,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            train=True,
        )
        episode_summary["episode"] = episode
        episode_logs.append(episode_summary)

        if episode == 1 or episode % 10 == 0 or episode == num_episodes:
            print(
                "Episode "
                f"{episode}/{num_episodes} | "
                f"reward={episode_summary['total_reward']:.3f} | "
                f"loss={episode_summary['average_loss']:.4f} | "
                f"epsilon={episode_summary['average_epsilon']:.4f}"
            )

    run_id = _timestamp()
    log_path = LOG_DIR / f"training_log_{run_id}.csv"
    model_path = MODEL_DIR / f"double_dqn_{run_id}.pt"
    config_path = LOG_DIR / f"experiment_config_{run_id}.json"

    _write_logs(log_path, episode_logs)
    agent.save(model_path)
    save_experiment_config(
        {
            "seed": seed,
            "num_episodes": num_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "config": {
                name: getattr(config_module, name)
                for name in dir(config_module)
                if name.isupper()
            },
        },
        config_path,
    )

    return {
        "agent": agent,
        "replay_buffer": replay_buffer,
        "logs": episode_logs,
        "log_path": log_path,
        "model_path": model_path,
        "config_path": config_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DDQN text-hint ITS agent.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )
    print(f"Saved training log: {result['log_path']}")
    print(f"Saved model: {result['model_path']}")
    print(f"Saved config snapshot: {result['config_path']}")
