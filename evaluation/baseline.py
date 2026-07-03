"""Baseline hint recommendation policies for evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomHintPolicy:
    """Uniformly sample one of the three hint actions."""

    action_dim: int = 3
    rng: random.Random = field(default_factory=random.Random)

    def select_action(self, state: np.ndarray) -> int:
        return self.rng.randrange(self.action_dim)


@dataclass
class RuleBasedHintPolicy:
    """Select hint strength from mastery, accuracy, difficulty, and attempts."""

    def select_action(self, state: np.ndarray) -> int:
        mastery = float(state[0])
        difficulty = float(state[1])
        recent_accuracy = float(state[2])
        normalized_attempts = float(state[4])

        if mastery < 0.4 or recent_accuracy < 0.34 or normalized_attempts > 0.67:
            return 0
        if mastery < 0.7 or difficulty >= 0.5 or recent_accuracy < 0.67:
            return 1
        return 2
