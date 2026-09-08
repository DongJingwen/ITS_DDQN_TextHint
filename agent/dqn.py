"""Conventional DQN; all infrastructure is shared with Double DQN."""

import torch

from agent.double_dqn import DoubleDQNAgent


class DQNAgent(DoubleDQNAgent):
    @torch.no_grad()
    def compute_td_targets(self, rewards: torch.Tensor, next_states: torch.Tensor,
                           dones: torch.Tensor) -> torch.Tensor:
        """DQN selects and evaluates the maximum using the target network."""
        next_values = self.target_network(next_states).max(dim=1).values
        return rewards + self.gamma * next_values * (1.0 - dones)
