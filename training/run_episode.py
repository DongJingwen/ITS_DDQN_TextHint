"""Run one training episode for the DDQN tutoring system."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.config import ACTION_DIM, BATCH_SIZE


def run_episode(
    environment: Any,
    agent: Any,
    replay_buffer: Any,
    batch_size: int = BATCH_SIZE,
    train: bool = True,
) -> dict[str, Any]:
    """Run one environment episode and optionally update the DDQN agent."""
    state = environment.reset()
    done = False

    total_reward = 0.0
    losses: list[float] = []
    epsilons: list[float] = []
    entropies: list[float] = []
    attempts: list[float] = []
    response_times: list[float] = []
    action_counts = {action: 0 for action in range(ACTION_DIM)}
    step_count = 0

    while not done:
        epsilon_count_before = len(agent.epsilon_history)
        action = agent.select_action(state)
        if len(agent.epsilon_history) > epsilon_count_before:
            epsilons.append(agent.epsilon_history[-1])
        if getattr(agent, "last_entropy", None) is not None:
            entropies.append(float(agent.last_entropy))

        next_state, reward, done, info = environment.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        if train and len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            losses.append(agent.update(batch))

        total_reward += reward
        attempts.append(float(info["attempts"]))
        response_times.append(float(info["response_time"]))
        action_counts[action] += 1
        step_count += 1
        state = next_state

    average_loss = sum(losses) / len(losses) if losses else 0.0
    average_epsilon = sum(epsilons) / len(epsilons) if epsilons else 0.0
    average_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    average_attempts = sum(attempts) / len(attempts) if attempts else 0.0
    average_response_time = (
        sum(response_times) / len(response_times) if response_times else 0.0
    )

    return {
        "total_reward": total_reward,
        "average_loss": average_loss,
        "average_epsilon": average_epsilon,
        "average_entropy": average_entropy,
        "average_attempts": average_attempts,
        "average_response_time": average_response_time,
        "action_0_count": action_counts[0],
        "action_1_count": action_counts[1],
        "action_2_count": action_counts[2],
        "steps": step_count,
    }
