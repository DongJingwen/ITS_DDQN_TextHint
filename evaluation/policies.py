"""Minimal evaluation contract: observation in, hint action out."""

try:
    from typing import Protocol
except ImportError:  # Python 3.7, supported by the existing project runtime.
    from typing_extensions import Protocol

import numpy as np
import torch

from agent.double_dqn import DoubleDQNAgent
from evaluation.baseline import RandomHintPolicy, RuleBasedHintPolicy


class HintPolicy(Protocol):
    def select_action(self, state: np.ndarray) -> int: ...


class FixedHintPolicy:
    """Always provide Partial Guidance (action 1 / level_2)."""

    def select_action(self, state: np.ndarray) -> int:
        return 1


class GreedyQPolicy:
    """Read-only greedy inference, without exploration/history/replay changes."""

    def __init__(self, agent: DoubleDQNAgent):
        self.agent = agent

    def select_action(self, state: np.ndarray) -> int:
        # Current QNetwork contains only Linear/ReLU, no mode-dependent layers.
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32,
                                     device=self.agent.device).unsqueeze(0)
            return int(self.agent.online_network(tensor).argmax(dim=1).item())
