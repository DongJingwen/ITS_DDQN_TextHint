"""Double DQN agent with dynamic entropy-based epsilon exploration."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.config import (
    ACTION_DIM,
    EPSILON_FIXED,
    EPSILON_MAX,
    EPSILON_MIN,
    EPSILON_MODE,
    GAMMA,
    LAMBDA_ENTROPY,
    LEARNING_RATE,
    SOFTMAX_TEMPERATURE,
    STATE_DIM,
    TARGET_UPDATE_FREQUENCY,
)


class QNetwork(nn.Module):
    """Small MLP for estimating Q-values over the three hint actions."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DoubleDQNAgent:
    """Double DQN agent for selecting text-hint support levels."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        epsilon_min: float = EPSILON_MIN,
        epsilon_max: float = EPSILON_MAX,
        epsilon_fixed: float = EPSILON_FIXED,
        epsilon_mode: str = EPSILON_MODE,
        lambda_entropy: float = LAMBDA_ENTROPY,
        softmax_temperature: float = SOFTMAX_TEMPERATURE,
        target_update_frequency: int = TARGET_UPDATE_FREQUENCY,
        device: str | torch.device | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_fixed = epsilon_fixed
        self.epsilon_mode = epsilon_mode
        self.lambda_entropy = lambda_entropy
        self.softmax_temperature = softmax_temperature
        self.target_update_frequency = target_update_frequency
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_history: list[float] = []
        self.epsilon_history: list[float] = []
        self.entropy_history: list[float] = []
        self.last_entropy: float | None = None
        self.update_count = 0
        self.current_episode = 1
        self.total_episodes = 1

        valid_modes = {"dynamic_entropy", "fixed", "linear_decay"}
        if self.epsilon_mode not in valid_modes:
            raise ValueError(f"epsilon_mode must be one of {sorted(valid_modes)}")

    def compute_dynamic_epsilon(self, q_values: torch.Tensor | np.ndarray) -> float:
        """Compute entropy-based epsilon from the current Q-value distribution."""
        if not torch.is_tensor(q_values):
            q_values = torch.as_tensor(q_values, dtype=torch.float32, device=self.device)
        else:
            q_values = q_values.detach().to(self.device).float()

        if q_values.dim() > 1:
            q_values = q_values.squeeze(0)

        temperature = max(self.softmax_temperature, 1e-8)
        probabilities = F.softmax(q_values / temperature, dim=-1)
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
        self.last_entropy = float(entropy.item())
        epsilon = self.epsilon_min + self.lambda_entropy * entropy.item()
        return float(np.clip(epsilon, self.epsilon_min, self.epsilon_max))

    def compute_fixed_epsilon(self) -> float:
        """Return a fixed epsilon value for ablation studies."""
        return float(np.clip(self.epsilon_fixed, self.epsilon_min, self.epsilon_max))

    def compute_linear_decay_epsilon(self) -> float:
        """Linearly decay epsilon from epsilon_max to epsilon_min over episodes."""
        if self.total_episodes <= 1:
            return self.epsilon_min
        progress = (self.current_episode - 1) / (self.total_episodes - 1)
        epsilon = self.epsilon_max - progress * (self.epsilon_max - self.epsilon_min)
        return float(np.clip(epsilon, self.epsilon_min, self.epsilon_max))

    def compute_epsilon(self, q_values: torch.Tensor | np.ndarray) -> float:
        """Compute epsilon according to the configured exploration mode."""
        if self.epsilon_mode == "dynamic_entropy":
            return self.compute_dynamic_epsilon(q_values)
        if self.epsilon_mode == "fixed":
            return self.compute_fixed_epsilon()
        if self.epsilon_mode == "linear_decay":
            return self.compute_linear_decay_epsilon()
        raise ValueError(f"Unsupported epsilon_mode: {self.epsilon_mode}")

    def set_training_progress(self, current_episode: int, total_episodes: int) -> None:
        """Set episode progress for linear-decay epsilon."""
        self.current_episode = max(1, int(current_episode))
        self.total_episodes = max(1, int(total_episodes))

    def select_action(self, state: np.ndarray | list[float]) -> int:
        """Select an action using dynamic epsilon-greedy exploration."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state_tensor).squeeze(0)
        self.online_network.train()

        epsilon = self.compute_epsilon(q_values)
        self.epsilon_history.append(epsilon)
        if self.last_entropy is not None:
            self.entropy_history.append(self.last_entropy)

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        return int(torch.argmax(q_values).item())

    def update(
        self,
        batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """Perform one Double DQN mini-batch update and return the loss."""
        states, actions, rewards, next_states, dones = batch

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.online_network(states_tensor).gather(1, actions_tensor).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.online_network(next_states_tensor), dim=1, keepdim=True)
            next_q_values = self.target_network(next_states_tensor).gather(1, next_actions).squeeze(1)
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1.0 - dones_tensor)

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.update_target_network()

        loss_value = float(loss.item())
        self.loss_history.append(loss_value)
        return loss_value

    def update_target_network(self) -> None:
        """Synchronize target network parameters from the online network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: str | Path) -> None:
        torch.save(self.online_network.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(state_dict)
        self.update_target_network()


if __name__ == "__main__":
    agent = DoubleDQNAgent(device="cpu")

    state = np.random.rand(STATE_DIM).astype(np.float32)
    action = agent.select_action(state)

    batch_size = 4
    batch = (
        np.random.rand(batch_size, STATE_DIM).astype(np.float32),
        np.random.randint(0, ACTION_DIM, size=batch_size),
        np.random.randn(batch_size).astype(np.float32),
        np.random.rand(batch_size, STATE_DIM).astype(np.float32),
        np.random.randint(0, 2, size=batch_size).astype(np.float32),
    )
    loss = agent.update(batch)

    print(f"Selected action: {action}")
    print(f"Last epsilon: {agent.epsilon_history[-1]:.4f}")
    print(f"Update loss: {loss:.4f}")
