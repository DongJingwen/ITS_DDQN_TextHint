"""Compare DDQN hint policy with simple baseline policies."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent.double_dqn import DoubleDQNAgent
from config.config import ACTION_DIM, MAX_STEPS_PER_EPISODE
from env.environment import TutoringEnvironment
from evaluation.baseline import RandomHintPolicy, RuleBasedHintPolicy


LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
COMPARISON_PATH = LOG_DIR / "baseline_comparison.csv"


class GreedyDDQNPolicy:
    """Evaluation wrapper that uses the trained DDQN policy without exploration."""

    def __init__(self, agent: DoubleDQNAgent) -> None:
        self.agent = agent

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.agent.device,
        ).unsqueeze(0)
        self.agent.online_network.eval()
        with torch.no_grad():
            q_values = self.agent.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def _latest_model_path() -> Path:
    model_files = sorted(MODEL_DIR.glob("double_dqn_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No DDQN model files found in {MODEL_DIR}")
    return model_files[-1]


def _evaluate_policy(
    policy_name: str,
    policy: Any,
    num_episodes: int,
    max_steps_per_episode: int,
    seed: int,
) -> dict[str, Any]:
    total_rewards: list[float] = []
    correct_values: list[float] = []
    attempts: list[float] = []
    response_times: list[float] = []
    action_counts = {action: 0 for action in range(ACTION_DIM)}

    for episode_index in range(num_episodes):
        environment = TutoringEnvironment(
            max_steps=max_steps_per_episode,
            seed=seed + episode_index,
        )
        state = environment.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = policy.select_action(state)
            next_state, reward, done, info = environment.step(action)

            episode_reward += reward
            correct_values.append(1.0 if info["correct"] else 0.0)
            attempts.append(float(info["attempts"]))
            response_times.append(float(info["response_time"]))
            action_counts[action] += 1
            state = next_state

        total_rewards.append(episode_reward)

    total_actions = sum(action_counts.values()) or 1

    return {
        "policy": policy_name,
        "episodes": num_episodes,
        "average_reward": sum(total_rewards) / len(total_rewards),
        "correct_after_hint_rate": sum(correct_values) / len(correct_values),
        "average_attempts": sum(attempts) / len(attempts),
        "average_response_time": sum(response_times) / len(response_times),
        "action_0_count": action_counts[0],
        "action_1_count": action_counts[1],
        "action_2_count": action_counts[2],
        "action_0_ratio": action_counts[0] / total_actions,
        "action_1_ratio": action_counts[1] / total_actions,
        "action_2_ratio": action_counts[2] / total_actions,
    }


def _write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "episodes",
        "average_reward",
        "correct_after_hint_rate",
        "average_attempts",
        "average_response_time",
        "action_0_count",
        "action_1_count",
        "action_2_count",
        "action_0_ratio",
        "action_1_ratio",
        "action_2_ratio",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def evaluate(
    model_path: str | Path | None = None,
    num_episodes: int = 20,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    seed: int = 100,
    output_path: str | Path = COMPARISON_PATH,
) -> list[dict[str, Any]]:
    """Evaluate Random, Rule-based, and trained DDQN policies."""
    model_path = Path(model_path) if model_path else _latest_model_path()
    output_path = Path(output_path)

    ddqn_agent = DoubleDQNAgent(device="cpu")
    ddqn_agent.load(model_path)

    policies = [
        ("random", RandomHintPolicy()),
        ("rule_based", RuleBasedHintPolicy()),
        ("double_dqn", GreedyDDQNPolicy(ddqn_agent)),
    ]

    results = [
        _evaluate_policy(
            policy_name=name,
            policy=policy,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
        )
        for name, policy in policies
    ]

    _write_results(results, output_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DDQN against hint baselines.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=COMPARISON_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    comparison_results = evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        output_path=args.output,
    )

    for result in comparison_results:
        print(
            f"{result['policy']}: "
            f"avg_reward={result['average_reward']:.3f}, "
            f"correct_rate={result['correct_after_hint_rate']:.3f}, "
            f"avg_attempts={result['average_attempts']:.3f}, "
            f"avg_response_time={result['average_response_time']:.3f}"
        )
    print(f"Saved comparison results: {args.output}")
