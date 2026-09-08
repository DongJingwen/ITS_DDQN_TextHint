"""Replay buffer for off-policy Double DQN training."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, NamedTuple

import numpy as np


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = capacity
        self.rng = rng if rng is not None else random
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )
        self.buffer.append(transition)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a mini-batch of transitions."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer with {len(self.buffer)} items."
            )

        transitions = self.rng.sample(list(self.buffer), batch_size)

        states = np.stack([transition.state for transition in transitions]).astype(np.float32)
        actions = np.array([transition.action for transition in transitions], dtype=np.int64)
        rewards = np.array([transition.reward for transition in transitions], dtype=np.float32)
        next_states = np.stack([transition.next_state for transition in transitions]).astype(
            np.float32
        )
        dones = np.array([transition.done for transition in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    replay_buffer = ReplayBuffer(capacity=5)

    for index in range(6):
        state = np.array([index, 0, 0, 0, 0, 0], dtype=np.float32)
        next_state = state + 1
        replay_buffer.push(
            state=state,
            action=index % 3,
            reward=float(index),
            next_state=next_state,
            done=index == 5,
        )

    batch = replay_buffer.sample(batch_size=3)

    print(f"Buffer length: {len(replay_buffer)}")
    print(f"States shape: {batch[0].shape}")
    print(f"Actions shape: {batch[1].shape}")
    print(f"Rewards shape: {batch[2].shape}")
    print(f"Next states shape: {batch[3].shape}")
    print(f"Dones shape: {batch[4].shape}")
